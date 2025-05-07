import contextlib
import gc
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, cast, override

import torch
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.reasoning_parser import ReasoningParser
from sglang.srt.server_args import ServerArgs

from eval_suite.client import BaseClientConfig
from eval_suite.client.base import BaseSamplingParams, ClientBase, Message
from eval_suite.client.utils import instruction_messages

logger = logging.getLogger(__name__)


@dataclass
class EvalServerArgs(ServerArgs):
    dp_size: int = torch.cuda.device_count()
    tp_size: int = 1
    disable_overlap_schedule: bool = True


class SGLangSamplingParams(BaseSamplingParams):
    # Core parameters
    min_p: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    temperature: float = 1.0
    max_new_tokens: int = 128
    min_new_tokens: int = 0

    # Penalizers
    repetition_penalty: float | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None

    # Constrained decoding
    json_schema: str | None = None
    regex: str | None = None
    ebnf: str | None = None

    # Other options
    n: int = 1
    spaces_between_special_tokens: bool = True
    no_stop_trim: bool = False
    continue_final_message: bool = False
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    custom_params: list[dict[str, Any]] | None = None


class SGLangClient(ClientBase[dict, SGLangSamplingParams, BaseClientConfig]):
    """
    Client for SGLang offline inference engine.

    enter: do nothing.
    exit: shutdown the engine, destroy process group, and clear GPU memory.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        *,
        config: BaseClientConfig = BaseClientConfig(),
        sampling_params: SGLangSamplingParams = SGLangSamplingParams(),
        reasoning_parser: Literal["deepseek-r1"] | None = None,
    ) -> None:
        super().__init__(
            model=server_args.model_path,
            sampling_params=sampling_params,
            config=config,
        )

        self._server_args = server_args
        self._reasoning_parser = (
            ReasoningParser(reasoning_parser)  #
            if reasoning_parser
            else None
        )

        # absolute path for `Engine` lazy loading
        model_path = Path(server_args.model_path).absolute()
        self._server_args.model_path = str(model_path)
        self._server_args.tokenizer_path = str(model_path)

    @cached_property
    def _engine(self):
        return Engine(server_args=self._server_args)

    @cached_property
    def _tokenizer(self):
        return get_tokenizer(
            tokenizer_name=self._server_args.model_path,
            trust_remote_code=True,
        )

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
        # do not cleanup if engine is never initialized
        if "_engine" not in self.__dict__:
            return

        import torch  # noqa

        self._engine.shutdown()
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _prepare_input(self, input: Sequence[dict[str, str]]) -> str:
        """Apply chat template manually"""

        return cast(
            str,
            self._tokenizer.apply_chat_template(
                list(input),
                tokenize=False,
                add_generation_prompt=True,
            ),
        )

    async def _generate(
        self,
        instructions: Iterable[str],
        *,
        system_prompt: str | None = None,
    ):
        msgs = [
            instruction_messages(
                instruction=instruction,
                system_prompt=system_prompt,
            )
            for instruction in instructions
        ]
        inputs = [self._prepare_input(msg) for msg in msgs]

        return cast(
            list[dict],
            await self._engine.async_generate(
                prompt=inputs,
                sampling_params=self._sampling_params.model_dump(),
            ),
        )

    @override
    async def generate(
        self,
        instructions: Iterable[str],
        *,
        system_prompt: str | None = None,
    ) -> Sequence[Message[dict] | None]:
        resps = await self._generate(
            instructions=instructions,
            system_prompt=system_prompt,
        )

        def to_message(resp: dict):
            match resp:
                case {"text": str(content)}:
                    content = content.strip()
                    # HACK prepend <think> tag in case of `generation_prompt` include `<think>` tag
                    if "</think>" in content and not content.startswith("<think>"):
                        content = f"<think>\n{content}"

                    reasoning_content: str | None = None
                    if parser := self._reasoning_parser:
                        reasoning_content, content = parser.parse_non_stream(content)

                    return Message(
                        content=content,
                        reasoning_content=reasoning_content,
                        generation=resp,
                    )
                case _:
                    logger.warning(f"Unexpected response format: {resp}")
                    return

        return [to_message(resp) for resp in resps]
