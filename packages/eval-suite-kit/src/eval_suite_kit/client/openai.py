import contextlib
from typing import override

from eval_suite_core.client import ClientConfig, OnlineClientBase, SamplingParamsBase
from openai import AsyncOpenAI


class OpenAISamplingParams(SamplingParamsBase):
    pass


class OpenAIClient(OnlineClientBase[OpenAISamplingParams]):
    client: AsyncOpenAI

    @override
    @classmethod
    @contextlib.asynccontextmanager
    async def init(
        cls,
        model: str,
        *,
        sampling_params: OpenAISamplingParams = OpenAISamplingParams(),
        config: ClientConfig = ClientConfig(),
    ):
        async with AsyncOpenAI() as client:
            yield cls(
                model=model,
                sampling_params=sampling_params,
                config=config,
                client=client,
            )
