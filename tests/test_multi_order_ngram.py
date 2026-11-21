
import math

import pytest

from aimusic.ml.grouped_ngram import GroupedNGramPrior
from aimusic.ml.multi_order_ngram import MultiOrderNGramPrior
from tests.test_grouped_ngram import state


@pytest.fixture
def model():
    pieces = [[
        state("Cmaj", "C"),
        state("G7", "C"),
        state("Amin", "C"),
        state("Cmaj", "C"),
        state("G7", "C"),
    ]] * 3

    models = []

    for order in (1, 2, 3):
        m = GroupedNGramPrior(
            key_size=12,
            chord_size=48,
            order=order,
            alpha=0.1,
        )
        m.fit(pieces)
        models.append(m)

    return MultiOrderNGramPrior(*models, strength=20)


def test_score_is_finite(model):
    score = model.logp_next(
        state("Cmaj", "C"),
        state("G7", "C"),
        0,
    )
    assert math.isfinite(score)


def test_score_between_component_scores(model):
    previous = state("Cmaj", "C")
    next_state = state("G7", "C")

    scores = [
        m.logp_next(previous, next_state, 0)
        for m in (
            model.unigram,
            model.bigram,
            model.trigram,
        )
    ]

    combined = model.logp_next(previous, next_state, 0)

    assert min(scores) <= combined <= max(scores)


def test_scalar_batch_equivalence(model):
    from aimusic.ml.dataset import build_transition_examples

    queries = build_transition_examples(
        [
            state("Cmaj", "C"),
            state("G7", "C"),
            state("Amin", "C"),
        ],
        history_len=2,
        future_hint_len=0,
    )

    batch = model.logp_next_batch(queries)
    scalar = tuple(
        model.logp_next(
            q.prev_state,
            q.next_state,
            q.time_index,
            q.context,
        )
        for q in queries
    )

    assert batch == pytest.approx(scalar)
