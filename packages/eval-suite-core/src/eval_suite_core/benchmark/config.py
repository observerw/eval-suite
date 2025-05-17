import os
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class EvalConfig(BaseModel):
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

    overwrite: bool = False
    """Whether to overwrite existing results when `stat_file` already exists"""

    use_cache: bool = True
    """Whether to use cached data"""

    with_timestamp: bool = False
    """Whether to add a timestamp to the output directory name"""

    concurrency: int = Field(default=os.cpu_count() or 64)
    """
    Concurrency for the evaluation process. This is the number of samples to evaluate in parallel. 
    
    For io-bounded evaluation, you may want to set this to a higher value for better performance.
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
