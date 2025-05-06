# BLEU Metric

## Introduction

BLEU (Bilingual Evaluation Understudy) is a metric for evaluating the quality of text that has been generated, particularly in machine translation and text generation tasks. It measures the similarity between the generated text and one or more reference texts by comparing the n-gram matches between them.

This metric is particularly useful for evaluating text generation tasks where the goal is to produce output that closely matches a reference or ground truth. BLEU scores range from 0 to 1, where higher values indicate greater similarity to the reference text.

## Schema

### Input

```python
class EvalInput(EvalInputBase):
    @property
    @abstractmethod
    def _reference(self) -> str:
        """The reference text to compare against."""
```

### Output

```python
class EvalOutput(EvalOutputBase):
    generation: str
    """Generated text to evaluate against the reference text."""
```

### Result

```python
class EvalResult(EvalResultBase):
    score: float
    """The BLEU score of the generated text compared to the reference text."""
```

## Statistics

### `ScoreStat`

The statistics object for BLEU score calculation uses the `ScoreStat` class, which is a general-purpose statistics class for score-based metrics.

```python
class ScoreStat(BaseModel):
    mean: float
    samples: int
    std: float
    min: float
    max: float
    median: float
```

- `mean`: The average BLEU score across all samples
- `samples`: The number of samples evaluated
- `std`: The standard deviation of the BLEU scores
- `min`: The minimum BLEU score
- `max`: The maximum BLEU score
- `median`: The median BLEU score

Example:

```json
{
    "mean": 0.75,
    "samples": 100,
    "std": 0.15,
    "min": 0.3,
    "max": 0.95,
    "median": 0.78
}
```

## Utilities

- `def bleu(reference: str, generation: str) -> float`: calculate the BLEU score between a reference text and a generated text.
  - `reference`: The reference text to compare against
  - `generation`: The generated text being evaluated
  - returns a float between [0, 1] representing the BLEU score

## Implementation

- `to_input`:
  - not required, but implementing classes must provide the `_reference` property
- `to_output`:
  - extracts the generated text from the model's response
  - returns an `EvalOutput` object with the generation
  - not required, default to using the generation as the output
- `to_result`:
  - calculates the BLEU score between the reference and the generated text
  - returns an `EvalResult` with the calculated score
  - not required, default to calculating the BLEU score using the `bleu` utility function
- `to_stat`:
  - uses the standard `ScoreStat` for score-based metrics
  - **required**
