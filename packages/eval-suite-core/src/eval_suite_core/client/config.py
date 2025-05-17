from abc import ABC

from pydantic import BaseModel


class SamplingParamsBase(BaseModel, ABC):
    model_config = {"frozen": True, "extra": "allow"}


class ClientConfig(BaseModel):
    batch_size: int = 256
    """
    Batch size for generation.

    It is recommended to set it according to the GPU memory, model size, and the except length of the generation.
    """
