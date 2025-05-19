from abc import ABC, abstractmethod

from pydantic import BaseModel


class CommunicatorBase(ABC):
    @abstractmethod
    async def print(self, message: str) -> None:
        """
        Print a message to the endpoint.
        """

    @abstractmethod
    async def request(self, message: str) -> str:
        """
        Request a message to the endpoint and return the response.
        """

    async def request_data[T: BaseModel](self, message: str, schema: T) -> T:
        """
        Request a message to the endpoint, validate the response with the schema, and return the response.
        """

        resp = await self.request(message)
        return schema.model_validate_json(resp)
