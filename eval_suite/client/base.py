import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from types import TracebackType
from typing import Any, override

from pydantic import BaseModel

from eval_suite.client.config import BaseClientConfig

logger = logging.getLogger(__name__)


class BaseSamplingParams(BaseModel):
    model_config = {"extra": "allow"}


class Message[Gen](BaseModel):
    content: str
    reasoning_content: str | None = None
    generation: Gen


class ClientBase[
    Gen: Any,
    Sampling: BaseSamplingParams,
    Config: BaseClientConfig,
](contextlib.AbstractContextManager, ABC):
    def __init__(
        self,
        model: str,
        *,
        sampling_params: Sampling = BaseSamplingParams(),
        config: Config = BaseClientConfig(),
    ) -> None:
        self._model = model
        self._sampling_params = sampling_params
        self._config = config

    @override
    def __enter__(self):
        return self

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ):
        return

    @property
    def model(self) -> str:
        return self._model

    @property
    def path_name(self) -> str:
        """Return a name for the model that can be used in file paths."""
        return self._model.replace("/", "_").replace(":", "_")

    @property
    def sampling_params(self) -> Sampling:
        return self._sampling_params

    @property
    def config(self) -> BaseClientConfig:
        return self._config

    @abstractmethod
    async def generate(
        self,
        instructions: Iterable[str],
        *,
        system_prompt: str | None = None,
    ) -> Sequence[Message[Gen] | None]: ...
