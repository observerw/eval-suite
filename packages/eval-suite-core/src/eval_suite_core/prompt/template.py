import base64
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from jinja2 import Template
from pydantic import BaseModel

from eval_suite_core.metric.item import ItemBase
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

type TextTemplateValue = str | Path | Template


@dataclass
class TextTemplate:
    value: TextTemplateValue

    def render(self, **kwargs) -> str:
        match self.value:
            case str() as text_template:
                template = create_template(text_template)
            case Path() as text_path:
                template = create_template(text_path.read_text(encoding="utf-8"))
            case Template() as template:
                pass

        rendered_text = template.render(**kwargs)

        return rendered_text


@dataclass
class ImageTemplate:
    value: str | Path | bytes

    def render(self, **kwargs) -> ImageUrl:
        match self.value:
            case str() as url_template:
                template = create_template(url_template)
                return ImageUrl(url=template.render(**kwargs))
            case Path() as image_path:
                image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
                return ImageUrl(url=image)
            case bytes() as image_bytes:
                image = base64.b64encode(image_bytes).decode("utf-8")
                return ImageUrl(url=image)


type TemplateValue = TextTemplateValue | list[TextTemplateValue | ImageTemplate]


def _render_content_template(
    value: TextTemplateValue | ImageTemplate, **kwargs
) -> TextContent | ImageContent:
    match value:
        case ImageTemplate() as image_template:
            image_url = image_template.render(**kwargs)
            return ImageContent(type="image_url", image_url=image_url)
        case text_template:
            text = TextTemplate(value=text_template).render(**kwargs)
            return TextContent(type="text", text=text)


class ChatTemplatePart(BaseModel):
    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    role: Role
    value: TemplateValue

    def format(self, **kwargs) -> ChatItem:
        match self.value:
            case list() as values:
                return ChatItem(
                    role=self.role,
                    content=[
                        _render_content_template(value=value, **kwargs)
                        for value in values
                    ],
                )
            case text_template:
                text = TextTemplate(value=text_template).render(**kwargs)
                return ChatItem(role=self.role, content=text)


def system_template(value: TemplateValue) -> ChatTemplatePart:
    return ChatTemplatePart(role="system", value=value)


def user_template(value: TemplateValue) -> ChatTemplatePart:
    return ChatTemplatePart(role="user", value=value)


def assistant_template(value: TemplateValue) -> ChatTemplatePart:
    return ChatTemplatePart(role="assistant", value=value)


class ChatTemplatePlaceholder(BaseModel):
    id: str = "history"


def template_placeholder(id: str = "history") -> ChatTemplatePlaceholder:
    return ChatTemplatePlaceholder(id=id)


@dataclass
class ChatTemplate[Item: ItemBase]:
    parts: list[ChatTemplatePart | ChatItem | ChatTemplatePlaceholder] = []
    formatters: list[FormatterBase | dict[str, Any]] = []

    @classmethod
    def compose(
        cls,
        *parts: ChatTemplatePart | ChatItem | ChatTemplatePlaceholder,
    ) -> Self:
        return cls(parts=[*parts])

    def __or__(self, other: FormatterBase) -> Self:
        self.formatters.append(other)
        return self

    def _format(self, item: Item, history: ChatSequence) -> Iterable[ChatItem]:
        variables: dict[str, Any] = {"history": history}

        for formatter in self.formatters:
            match formatter:
                case FormatterBase() as fmt:
                    variables[fmt.name] = fmt.provide(item=item, history=history)
                case dict() as fmt_dict:
                    variables.update(fmt_dict)

        for part in self.parts:
            match part:
                case ChatTemplatePart() as template_part:
                    yield template_part.format(**variables)
                case ChatTemplatePlaceholder() as placeholder:
                    sequence = variables.get(placeholder.id, [])
                    if not isinstance(sequence, Sequence):
                        raise ValueError(
                            f"Expected a ChatSequence for placeholder '{placeholder.id}', "
                            f"got {type(sequence).__name__} instead."
                        )
                    yield from sequence
                case chat_item:
                    yield chat_item

    def format(self, item: Item, history: ChatSequence) -> ChatSequence:
        return [*self._format(item=item, history=history)]
