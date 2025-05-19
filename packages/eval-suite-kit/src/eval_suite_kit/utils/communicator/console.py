from typing import override

from eval_suite_kit.utils.communicator.base import CommunicatorBase


class ConsoleCommunicator(CommunicatorBase):
    @override
    async def print(self, message: str) -> None:
        print(message)

    @override
    async def request(self, message: str) -> str:
        print(message)
        return input("Response: ")
