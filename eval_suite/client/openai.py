import asyncio as aio
import logging
from collections.abc import Iterable, Sequence
from functools import cached_property
from types import TracebackType
from typing import Self, cast, override

import docker
from openai import AsyncOpenAI, OpenAI, OpenAIError
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import SecretStr
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_fixed,
    wait_random_exponential,
)

from eval_suite.client import BaseClientConfig
from eval_suite.client.base import BaseSamplingParams, ClientBase, Message
from eval_suite.client.utils import instruction_messages

logger = logging.getLogger(__name__)


class OpenAISamplingParams(BaseSamplingParams):
    pass


@retry(
    retry=retry_if_exception_type((OpenAIError, ValueError)),
    wait=wait_fixed(10),
    stop=stop_after_attempt(60),
    after=after_log(logger, logging.INFO),
)
def wait_for_openai_server(
    model: str,
    *,
    api_key: SecretStr | None = None,
    base_url: str | None = None,
    timeout: int = 10,
):
    """Wait for a specific model to be ready on the OpenAI server"""

    logger.info(f"Waiting for OpenAI server to be ready for model: {model}")

    serving_models = OpenAI(
        api_key=api_key.get_secret_value() if api_key else None,
        base_url=base_url,
    ).models.list(timeout=timeout)

    if all(serving_model.id != model for serving_model in serving_models):
        raise ValueError(f"Model {model} is not available at {base_url}")


class OpenAIClient(ClientBase[ChatCompletion, OpenAISamplingParams, BaseClientConfig]):
    def __init__(
        self,
        model: str,
        *,
        config: BaseClientConfig = BaseClientConfig(),
        sampling_params: OpenAISamplingParams = OpenAISamplingParams(),
        api_key: SecretStr | None = None,
        base_url: str | None = None,
        docker_container_id: str | None = None,
    ):
        super().__init__(
            model,
            sampling_params=sampling_params,
            config=config,
        )

        self._docker_container_id = docker_container_id

        self._api_key = api_key
        self._base_url = base_url
        self._client = AsyncOpenAI(
            api_key=api_key.get_secret_value() if api_key else None,
            base_url=base_url,
        )

    @cached_property
    def _container(self):
        if id := self._docker_container_id:
            return docker.from_env().containers.get(id)

    @override
    def __enter__(self) -> Self:
        if container := self._container:
            container.start()

        wait_for_openai_server(
            model=self._model,
            api_key=self._api_key,
            base_url=self._base_url,
        )

        return self

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ):
        if container := self._container:
            container.stop()

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10),
        retry=retry_if_not_exception_type(aio.CancelledError),
        after=after_log(logger, logging.WARNING),
    )
    async def _generate(
        self,
        instruction: str,
        *,
        system_prompt: str | None = None,
    ):
        msg = instruction_messages(instruction=instruction, system_prompt=system_prompt)

        match await self._client.chat.completions.create(
            model=self._model,
            messages=cast(list[ChatCompletionMessageParam], msg),
            **self.sampling_params.model_dump(),
        ):
            case ChatCompletion(choices=[Choice(message=msg)]) as comp if (
                content := msg.content
            ):
                reasoning_content = (msg.model_extra or {}).get(
                    "reasoning_content",
                    None,
                )
                assert isinstance(reasoning_content, str | None)
                return Message(
                    content=content,
                    reasoning_content=reasoning_content,
                    generation=comp,
                )
            case _:
                return

    async def generate(
        self,
        instructions: Iterable[str],
        *,
        system_prompt: str | None = None,
    ) -> Sequence[Message[ChatCompletion] | None]:
        async with aio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self._generate(instruction, system_prompt=system_prompt))
                for instruction in instructions
            ]

        return [task.result() for task in tasks]
