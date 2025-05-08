from abc import abstractmethod
from pathlib import Path

from eval_suite import (
    BaseEvalConfig,
    BenchmarkBase,
    EvalInputBase,
    EvalOutputBase,
    EvalResultBase,
    EvalStatBase,
)
from eval_suite.client.base import Message
from eval_suite_kit.metrics import score


class EvalInput(EvalInputBase):
    @property
    @abstractmethod
    def _reference(self) -> str:
        """The reference text to compare against."""


class EvalOutput(EvalOutputBase):
    generation: str
    """Generated text to evaluate against the reference text."""


class EvalResult(EvalResultBase):
    score: float
    """The BLEU score of the generated text compared to the reference text."""


def bleu(reference: str, generation: str) -> float:
    """
    Calculate the BLEU score from scratch, do not using any third-party libraries.
    """

    # Tokenize the reference and generation texts
    reference_tokens = reference.split()
    generation_tokens = generation.split()

    # Calculate the number of n-grams in the reference
    reference_ngrams = {}
    for i in range(len(reference_tokens)):
        for j in range(1, 5):
            if i + j <= len(reference_tokens):
                ngram = tuple(reference_tokens[i : i + j])
                if ngram not in reference_ngrams:
                    reference_ngrams[ngram] = 0
                reference_ngrams[ngram] += 1

    # Calculate the number of n-grams in the generation
    generation_ngrams = {}
    for i in range(len(generation_tokens)):
        for j in range(1, 5):
            if i + j <= len(generation_tokens):
                ngram = tuple(generation_tokens[i : i + j])
                if ngram not in generation_ngrams:
                    generation_ngrams[ngram] = 0
                generation_ngrams[ngram] += 1

    # Calculate precision
    precision = {}
    for ngram, count in generation_ngrams.items():
        if ngram in reference_ngrams:
            precision[ngram] = min(count, reference_ngrams[ngram]) / count

    # Calculate BLEU score
    bleu_score = 1.0
    for ngram, p in precision.items():
        bleu_score *= p ** (1 / len(precision))

    return bleu_score


BleuStat = score.Stat


class BleuBenchmark[Input: EvalInput, Stat: EvalStatBase](
    BenchmarkBase[Input, EvalOutput, EvalResult, Stat, BaseEvalConfig]
):
    def to_output(self, generation: Message, input: Input) -> EvalOutput:
        return EvalOutput(generation=generation.content)

    def to_result_sync(
        self,
        eval_path: Path,
        input: Input,
        output: EvalOutput,
    ) -> EvalResult:
        reference = input._reference
        score = bleu(reference, output.generation)
        return EvalResult(score=score)
