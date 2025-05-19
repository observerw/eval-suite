from collections.abc import Iterable, Sequence
from typing import override

from eval_suite_core.client import Message, OfflineClientBase
from eval_suite_core.client.config import SamplingParamsBase
from eval_suite_core.prompt.schema import ChatSequence


class DummySamplingParams(SamplingParamsBase):
    pass


class OfflineDummyClient(OfflineClientBase[DummySamplingParams]):
    """Dummy client for testing purposes."""

    model: str = "dummy"
    sampling_params: DummySamplingParams = DummySamplingParams()

    @override
    def generate(
        self, seq_batch: Iterable[ChatSequence], *, system_prompt: str | None = None
    ) -> Sequence[Message | None]:
        return []
