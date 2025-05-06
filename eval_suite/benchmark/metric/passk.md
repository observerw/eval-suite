# Pass@k Metric

## Introduction

Pass@k is a metric for evaluating the performance of code generation models. It measures the probability that at least one solution is correct when the model generates multiple solutions. Specifically, if a model generates n samples with c of them passing the test, Pass@k is the probability that at least one sample passes when k samples are randomly selected from the n samples.

This metric is particularly useful for evaluating code generation tasks, especially in scenarios where the success rate of a single generation might be low, but the correct answer can be obtained through multiple sampling attempts.

## Schema

### Output

```python
class EvalOutput(EvalOutputBase):
    code: str
    """The completed code (including imports and unit tests) to evaluate"""
```

### Result

```python
class EvalResult(EvalResultBase):
    passed: bool = True
    """Whether the code passed the unit test"""

    type: str = BaseEvalResultType.success
    """Result type, default to success"""
```

## Statistics

### `PassKStat`

The statistics object for basic pass@k calculation.

```python
class PassKStat(BaseModel):
    k: int
    pass_n: dict[str, float]
```

- `k`: the number of passes
- `pass_n`: a dictionary with keys `pass@1`, `pass@2`, ..., `pass@k` and values in the range [0, 1] indicating the pass rate for each pass.

The statistics object provides a `from_groups` class method to generate statistics from evaluation result groups:

```python
@classmethod
def from_groups(cls, groups: EvalResultGroups[EvalResult], k: int) -> Self:
    """Generate Pass@k statistics from evaluation result groups"""
```

Example:

```json
{
    "k": 5,
    "pass_n": {
        "pass@1": 0.5,
        "pass@2": 0.7,
        "pass@3": 0.8,
        "pass@4": 0.9,
        "pass@5": 1.0
    }
}
```

## Extra Config

| Name        | Type                             | Default | Description                                              |
| ----------- | -------------------------------- | ------- | -------------------------------------------------------- |
| `k`         | `int` greater than or equal to 1 | 5       | The number of passes to evaluate.                        |
| `n_samples` | `int`                            | 10      | Number of samples to generate for each evaluation input. |

### Requirements

- the `n_samples` must be greater than `k` (to calculate a meaningful pass@k, `k` must be less than `n_samples`)

## Utilities

- `def pass_k(n: int, c: int, k: int) -> float`: calculate the pass rate for `n` samples and `c` correct samples for `k` passes.
  - `n`: total number of samples
  - `c`: number of samples that pass the test
  - `k`: number of passes to consider
  - returns a float between [0, 1] representing the pass@k probability

## Implementation

- `to_input`:
  - not required
- `to_output`:
  - extract and compose the code
  - not required, default is to use the generation as the code
- `to_result*`:
  - extract and compose the code
  - **required**
- `to_stat`:
  - **required**
