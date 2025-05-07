import os
from pathlib import Path

from pydantic import BaseModel, Field


class BenchmarkConfig(BaseModel):
    """Base configuration for benchmarking"""

    stat_file: Path = Path("stat.json")
    """
    Relative json file path to save the statistics of the evaluation. 
    
    Will be used to determine whtether the evaluation is finished or not.
    """

    results_file: Path | None = Path("results.json")
    """Relative json file path to save the results of the evaluation"""

    config_file: Path | None = Path("config.json")
    """Relative json file path to save the configuration used for the evaluation"""

    overwrite: bool = False
    """Whether to overwrite existing results when `stat_file` already exists"""

    use_cache: bool = True
    """Whether to use cached data"""

    with_timestamp: bool = False
    """Whether to add a timestamp to the output directory name"""

    eval_batch_size: int = Field(default=os.cpu_count() or 64)
    """
    Batch size for the evaluation process. This is the number of samples to evaluate in parallel. 
    
    For io-bounded evaluation, you may want to set this to a higher value for better performance.
    """

    overlap: bool = True
    """
    Whether to overlap the generation and evaluation processes. 
    
    In rare cases where the evaluation and generation require same resources (e.g. GPU), this can be set to `False` to avoid resource contention.

    The framework will use this information to optimize the evaluation process.
    """


class BaseEvalConfig(BaseModel):
    """Base configuration for evaluation"""

    n_samples: int = 1
    """Number of samples for each item"""

    max_n_samples: int | None = Field(
        default_factory=lambda data: data["n_samples"] * 2
    )
    """
    Maximum number of samples for each item until at least `n_samples` of valid results are reached. 
    
    `None` means no limit (not recommended).
    """

    system_prompt: str | None = None
    """System prompt to use for the generation"""
