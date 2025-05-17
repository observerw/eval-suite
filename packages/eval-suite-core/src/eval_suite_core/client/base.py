import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, override

from pydantic import BaseModel

from eval_suite_core import prompt
from eval_suite_core.client.config import ClientConfig
from eval_suite_core.prompt.schema import ChatContent, ChatSequence

logger = logging.getLogger(__name__)


class SamplingParamsBase(BaseModel, ABC):
    model_config = {"frozen": True, "extra": "allow"}


class Message(BaseModel):
    content: str
    reasoning_content: str | None = None
    generation: dict[str, Any] | None = None


class _ClientBase[P: SamplingParamsBase](contextlib.AbstractContextManager, ABC):
    def __init__(
        self,
        model: str,
        sampling_params: P,
        *,
        config: ClientConfig = ClientConfig(),
    ) -> None:
        self._model = model
        self._sampling_params = sampling_params
        self._config = config

    @override
    def __enter__(self):
        return self

    @override
    def __exit__(self, *exc_details):
        return

    @property
    def model(self) -> str:
        return self._model

    @property
    def path_name(self) -> str:
        """Return a name for the model that can be used in file paths."""
        return self._model.replace("/", "_").replace(":", "_")

    @property
    def sampling_params(self) -> P:
        return self._sampling_params

    @property
    def config(self) -> ClientConfig:
        return self._config


class OnlineClientBase[P: SamplingParamsBase](_ClientBase[P]):
    """
    Client for streaming inference.

    This client is intended to be used with API-based inference server like Online API, or self-hosted OpenAI-Compatible server.
    """

    @abstractmethod
    async def generate(self, sequence: ChatSequence) -> Message | None: ...

    async def instruct(
        self, instruction: ChatContent, *, system_prompt: str | None = None
    ) -> Message | None:
        seq = [prompt.user(instruction)]
        if system_prompt:
            seq.insert(0, prompt.system(system_prompt))

        return await self.generate(seq)


class OfflineClientBase[P: SamplingParamsBase](_ClientBase[P]):
    """
    Client for batch inference.

    This client is intended to be used with offline inference engine, like vLLM `LLM` class, or SGLang `Engine` class.
    """

    @abstractmethod
    def generate(
        self, seq_batch: Iterable[ChatSequence], *, system_prompt: str | None = None
    ) -> Sequence[Message | None]: ...

    def instruct(
        self, instructions: Iterable[ChatContent], *, system_prompt: str | None = None
    ) -> Sequence[Message | None]:
        seq_batch = [[prompt.user(instruction)] for instruction in instructions]
        if system_prompt:
            for seq in seq_batch:
                seq.insert(0, prompt.system(system_prompt))

        return self.generate(seq_batch)
