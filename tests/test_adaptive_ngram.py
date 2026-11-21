
import math

import pytest

from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.ml.adaptive_ngram import AdaptiveNGramPrior
from aimusic.ml.grouped_ngram import GroupedNGramPrior
from aimusic.ml.ngram_prior import NGramConfig, NGramPrior
from tests.test_grouped_ngram import state


@pytest.fixture
def adaptive():
    vocab = DEFAULT_VOCABULARIES

    individual = NGramPrior.from_vocabularies(
        vocab,
        config=NGramConfig(
            order=3,
            alpha=0.1,
            streams=("chord", "key"),
        ),
    )

    grouped = GroupedNGramPrior.from_vocabularies(
        vocab, order=3, alpha=0.1
    )

    piece = [
        state("Cmaj", "C"),
        state("G7", "C"),
        state("Amin", "C"),
        state("Cmaj", "C"),
        state("G7", "C"),
        state("Amin", "C"),
    ]

    individual.fit([piece] * 3)
    grouped.fit([piece] * 3)

    return AdaptiveNGramPrior(
        individual, grouped, strength=20
    )


def test_adaptive_score_is_finite(adaptive):
    score = adaptive.logp_next(
        state("Cmaj", "C"),
        state("G7", "C"),
        0,
    )

    assert math.isfinite(score)


def test_adaptive_score_is_between_components(adaptive):
    previous = state("Cmaj", "C")
    next_state = state("G7", "C")

    individual_score = adaptive.individual.logp_next(
        previous, next_state, 0
    )

    grouped_score = adaptive.grouped.logp_next(
        previous, next_state, 0
    )

    adaptive_score = adaptive.logp_next(
        previous, next_state, 0
    )

    assert min(individual_score, grouped_score) <= adaptive_score
    assert adaptive_score <= max(individual_score, grouped_score)


def test_adaptive_weight(adaptive):
    observed = adaptive.grouped_weight(
        state("Cmaj", "C")
    )

    assert 0 < observed < 1

    # Larger strength means more caution about grouped evidence.
    cautious = AdaptiveNGramPrior(
        adaptive.individual,
        adaptive.grouped,
        strength=100,
    )

    assert cautious.grouped_weight(
        state("Cmaj", "C")
    ) < observed
