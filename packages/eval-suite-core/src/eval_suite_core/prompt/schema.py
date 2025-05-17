from collections.abc import Sequence
from typing import Literal, TypedDict


class TextContent(TypedDict, total=False):
    type: Literal["text"]
    text: str


class ImageUrl(TypedDict):
    url: str


class ImageContent(TypedDict, total=False):
    type: Literal["image_url"]
    image_url: ImageUrl


type Role = Literal["system", "user", "assistant"]

type ChatContent = str | list[TextContent | ImageContent]


class ChatItem(TypedDict):
    role: Role
    content: str | list[TextContent | ImageContent]


type ChatSequence = Sequence[ChatItem]


def system(content: ChatContent) -> ChatItem:
    return ChatItem(role="system", content=content)


def user(content: ChatContent) -> ChatItem:
    return ChatItem(role="user", content=content)


def assistant(content: ChatContent) -> ChatItem:
    return ChatItem(role="assistant", content=content)
