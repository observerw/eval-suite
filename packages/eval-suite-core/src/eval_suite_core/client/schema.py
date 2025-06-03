from pydantic import BaseModel


class Message[T](BaseModel):
    content: str
    """ The content of the response."""

    reasoning_content: str | None = None
    """(Reasoning model only) The content of the reasoning trace."""

    generation: T | None = None
    """The raw data of the generation, if available."""
