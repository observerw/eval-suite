from typing import Any, Self, override

import numpy as np
from pydantic import computed_field

from eval_suite.benchmark import EvalResultBase, EvalResultGroups, EvalStatBase


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


class EvalResult(EvalResultBase):
    reference: str
    generation: str

    @computed_field
    def bleu_score(self) -> float:
        """Calculate the BLEU score for the result."""

        return bleu(self.reference, self.generation)

    @override
    def model_post_init(self, context: Any) -> None:
        # init on creation
        _ = self.bleu_score


class Stat[Result: EvalResult](EvalStatBase):
    avg_bleu_score: float

    @classmethod
    def from_groups(cls, groups: EvalResultGroups[Result]) -> Self:
        """Calculate the average BLEU score from the groups of results."""

        bleu_scores = [
            bleu(result.reference, result.generation)
            for group in groups.values()
            for result in group
        ]
        avg_bleu_score = float(np.mean(bleu_scores)) if bleu_scores else 0.0
        return cls(avg_bleu_score=avg_bleu_score)
