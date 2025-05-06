from collections.abc import Iterable, Sequence

from eval_suite.client import ClientBase
from eval_suite.client.base import Message


class VLLMClient(ClientBase):
    async def generate(
        self,
        instructions: Iterable[str],
        *,
        system_prompt: str | None = None,
    ) -> Sequence[Message | None]:
        raise NotImplementedError
