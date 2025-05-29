from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel


class Schema(BaseModel):
    model_config = {"frozen": True, "extra": "allow"}


class TextContent(Schema):
    type: Literal["text"] = "text"
    text: str


class ImageUrl(Schema):
    url: str


class ImageContent(Schema):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


type Role = Literal["system", "user", "assistant"]

type ChatContent = str | list[TextContent | ImageContent]


class ChatItem(Schema):
    role: Role
    content: str | list[TextContent | ImageContent]


type ChatSequence = Sequence[ChatItem]


def system(content: ChatContent) -> ChatItem:
    return ChatItem(role="system", content=content)


def user(content: ChatContent) -> ChatItem:
    return ChatItem(role="user", content=content)


def assistant(content: ChatContent) -> ChatItem:
    return ChatItem(role="assistant", content=content)
