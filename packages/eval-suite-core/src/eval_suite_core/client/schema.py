from typing import Any

from pydantic import BaseModel


class Message(BaseModel):
    content: str
    reasoning_content: str | None = None
    generation: dict[str, Any] | None = None
