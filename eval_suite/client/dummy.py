from collections.abc import Iterable, Sequence

from eval_suite.client import BaseClientConfig, ClientBase
from eval_suite.client.base import BaseSamplingParams, Message


class DummyClient(ClientBase[dict, BaseSamplingParams, BaseClientConfig]):
    """Dummy client for testing purposes."""

    def __init__(
        self,
    ) -> None:
        super().__init__(
            model="dummy",
            config=BaseClientConfig(),
            sampling_params=BaseSamplingParams(),
        )

    async def generate(
        self,
        instructions: Iterable[str],
        *,
        system_prompt: str | None = None,
    ) -> Sequence[Message[dict] | None]:
        return [
            Message(
                content=instruction,
                generation={},
            )
            for instruction in instructions
        ]
