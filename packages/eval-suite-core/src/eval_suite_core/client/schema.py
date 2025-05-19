from typing import Any

from pydantic import BaseModel


class Message(BaseModel):
    content: str
    """ The content of the response."""

    reasoning_content: str | None = None
    """(Reasoning model only) The content of the reasoning trace."""

    generation: dict[str, Any] | None = None
    """The raw data of the generation, if available."""
