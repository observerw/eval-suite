from collections.abc import Sequence
from typing import Any, Self

from jinja2 import Template
from pydantic import BaseModel

from eval_suite_core.metric.item import EvalItemBase
from eval_suite_core.prompt.formatter import FormatterBase
from eval_suite_core.prompt.schema import (
    ChatItem,
    ChatSequence,
    ImageContent,
    ImageUrl,
    Role,
    TextContent,
)
from eval_suite_core.prompt.utils import create_template

type TemplateValue = str | Template | list[str | Template | ImageUrl]


class ChatTemplatePart(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    role: Role
    value: Template | list[Template | ImageUrl]

    @classmethod
    def build(cls, role: Role, value: TemplateValue) -> Self:
        match value:
            case str() as template_string:
                part = create_template(template_string)
            case list() as values:
                part = [
                    create_template(value) if isinstance(value, str) else value
                    for value in values
                ]
            case _:
                raise ValueError(f"Invalid template value: {value}.")

        return cls(role=role, value=part)

    def format(self, variables: dict[str, Any]) -> ChatItem:
        match self.value:
            case Template() as template:
                content = template.render(variables)
            case list() as values:
                content = [
                    TextContent(text=value.render(variables))
                    if isinstance(value, Template)
                    else ImageContent(image_url=value)
                    for value in values
                ]

        return ChatItem(role=self.role, content=content)


def system(value: TemplateValue) -> ChatTemplatePart:
    return ChatTemplatePart.build(role="system", value=value)


def user(value: TemplateValue) -> ChatTemplatePart:
    return ChatTemplatePart.build(role="user", value=value)


def assistant(value: TemplateValue) -> ChatTemplatePart:
    return ChatTemplatePart.build(role="assistant", value=value)


class ChatTemplatePlaceholder(BaseModel):
    id: str

    def format(self, variables: dict[str, Any]) -> Sequence[ChatItem]:
        if self.id not in variables:
            return []

        value = variables[self.id]
        return value


class ChatTemplate[Item: EvalItemBase](BaseModel):
    parts: list[ChatTemplatePart] = []
    formatters: list[FormatterBase] = []

    @classmethod
    def compose(cls, *parts: ChatTemplatePart) -> Self:
        return cls(parts=list(parts))

    def __rshift__(self, formatter: FormatterBase) -> Self:
        self.formatters.append(formatter)
        return self

    def format(self, item: Item, history: ChatSequence) -> Sequence[ChatItem]:
        variables: dict[str, Any] = {
            "history": history,
        }

        for formatter in self.formatters:
            variables.update(formatter.provide(item=item, history=history))

        return [part.format(variables=variables) for part in self.parts]
