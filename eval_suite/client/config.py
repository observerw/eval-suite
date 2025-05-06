from eval_suite.benchmark.schema import BaseModel


class BaseClientConfig(BaseModel):
    batch_size: int = 256
    """
    Batch size for generation.

    It is recommended to set it according to the GPU memory, model size, and the except length of the generation.
    """
