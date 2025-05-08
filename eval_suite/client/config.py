from eval_suite.metric.schema import BaseModel


class BaseClientConfig(BaseModel):
    batch_size: int = 256
    """
    Batch size for generation.

    It is recommended to set it according to the GPU memory, model size, and the except length of the generation.
    """

    stream_generation: bool = False
    """
    Whether to enable streaming generation.

    When enabled, we'll call `generate` with only one input at a time, and maintain `batch_size` inputs running at the same time.

    This is useful for real-time generation, e.g. API-based generation, but not for offline batch generation.
    """
