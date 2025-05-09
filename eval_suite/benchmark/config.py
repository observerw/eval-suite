import os
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class BenchmarkConfig(BaseModel):
    """Base configuration for benchmarking"""

    stat_file: Path = Path("stat.json")
    """
    Relative json file path to save the statistics of the evaluation. 
    
    Will be used to determine whtether the evaluation is finished or not.
    """

    results_file: Path = Path("results.json")
    """Relative json file path to save the results of the evaluation"""

    results_dir: Path = Path("results")
    """Relative directory path to save all the results of the evaluation"""

    config_file: Path = Path("config.json")
    """Relative json file path to save the configuration used for the evaluation"""

    output_organize: Literal["model-first", "benchmark-first"] = "model-first"
    """
    How to organize the output directory.

    - `model-first`: The output directory will be `<base_path>/<model_name>/<benchmark_name>`.
    - `benchmark-first`: The output directory will be `<base_path>/<benchmark_name>/<model_name>`.
    """

    exception_level: Literal["strict", "standard", "ignore"] = "standard"
    """
    TODO The level of exception handling.

    Strict level will raise exceptions early, but requires you to handle corner cases manually; Ignore level guarantees that the evaluation will continue, but problems may be hidden and lead to unexpected results.

    - `strict`: Any unexpected exception will be raised and the evaluation will be stopped.
    - `standard`: Some exceptions will be raised while most will be recorded as results.
    - `ignore`: All exceptions will be ignored and the evaluation will continue.
    """

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

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.max_n_samples and self.max_n_samples < self.n_samples:
            raise ValueError(
                "`max_n_samples` must be greater than or equal to `n_samples`"
            )

        return self
