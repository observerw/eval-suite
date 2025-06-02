import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

from pydantic import BaseModel

from eval_suite_core import prompt
from eval_suite_core.client.config import ClientConfig, SamplingParamsBase
from eval_suite_core.client.schema import Message
from eval_suite_core.prompt.schema import ChatContent, ChatSequence

logger = logging.getLogger(__name__)


class AnyClient[P: SamplingParamsBase](BaseModel, ABC):
    model: str
    sampling_params: P
    config: ClientConfig = ClientConfig()

    @contextlib.contextmanager
    def init(self):
        yield self

    @property
    def path_name(self) -> str:
        """Return a name for the model that can be used in file paths."""
        return self.model.replace("/", "_").replace(":", "_")


class OnlineClientBase[P: SamplingParamsBase](AnyClient[P]):
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


class OfflineClientBase[P: SamplingParamsBase](AnyClient[P]):
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
