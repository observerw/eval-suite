import re
from typing import Self, overload

from pydantic import BaseModel, RootModel

_IDENTIFIER_RE = re.compile(
    r"(?P<lang>[\w-]+)(?:\s*\[(?P<group>.+?)\])?(?:\s*\{(?P<id>.+?)\})?"
)
_CODEBLOCK_RE = re.compile(
    r"(`{3,})(?P<identifier>.*?)\n(?P<code>.*?)\n\1",
    re.DOTALL,
)


class CodeResult(BaseModel):
    model_config = {"frozen": True}

    lang: str
    code: str
    group: str | None
    id: str | None

    @classmethod
    def from_match(cls, match: re.Match[str]) -> Self | None:
        identifier = match.group("identifier").strip()
        if not (identifier_match := _IDENTIFIER_RE.match(identifier)):
            return

        return cls(
            lang=identifier_match.group("lang"),
            group=identifier_match.group("group"),
            id=identifier_match.group("id"),
            code=match.group("code").strip(),
        )


GroupResults = list[CodeResult]
GroupLookup = dict[str, GroupResults]
LangLookup = dict[str, GroupLookup]


class CodeResults(RootModel):
    model_config = {"frozen": True}

    root: LangLookup

    @overload
    def get(
        self,
        lang: str,
        *,
        id: str,
        group: str | None = ...,
    ) -> CodeResult | None: ...

    @overload
    def get(
        self,
        lang: str,
        *,
        group: str | None = ...,
        id: None = ...,
    ) -> list[CodeResult]: ...

    def get(
        self,
        lang: str,
        *,
        group: str | None = None,
        id: str | None = None,
    ) -> list[CodeResult] | CodeResult | None:
        lang_group = self.root.get(lang, {})
        match (group, id):
            case (None, None):
                return [
                    result
                    for group_results in lang_group.values()
                    for result in group_results
                ]
            case (group, None):
                return lang_group.get(group, [])
            case (group, id):
                group_results = lang_group.get(group or "default", [])
                result = next((r for r in group_results if r.id == id), None)
                return result


def extract_code(text: str) -> CodeResults:
    results: LangLookup = {}

    for match in _CODEBLOCK_RE.finditer(text):
        if not (result := CodeResult.from_match(match)):
            continue

        lang = result.lang
        group = result.group or "default"

        results.setdefault(lang, {}).setdefault(group, []).append(result)

    return CodeResults(root=results)


_BOXED_RE = re.compile(r"\\boxed{(?P<result>.*?)}", re.DOTALL)


def extract_boxed(text: str) -> str | None:
    if not (match := _BOXED_RE.search(text)):
        return

    return match.group("result").strip()
