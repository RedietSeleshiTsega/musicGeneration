
"""Additional regression tests requested during NGramPrior review."""

from dataclasses import replace
import math

import pytest

from aimusic.core.config import StyleConfig
from aimusic.core.core_types import Layer
from aimusic.core.vocab import build_default_vocabularies
from aimusic.ml.ngram_prior import NGramConfig, NGramPrior
from aimusic.planning.graph import _build_prior_context

from tests.ngram_test_helpers import state


@pytest.fixture
def vocab():
    return build_default_vocabularies(StyleConfig())


def test_untrained_model_uses_uniform_distribution(vocab):
    """With no counts, add-alpha smoothing gives 1/(V+1)."""

    model = NGramPrior.from_vocabularies(
        vocab,
        config=NGramConfig(
            order=3,
            alpha=0.1,
            streams=("chord",),
        ),
    )

    size = model.vocabulary_sizes["chord"]
    expected = 1.0 / (size + 1)

    for token in range(size + 1):
        probability = model.probability(
            "chord",
            token,
            history=(0, 1),
        )

        assert math.isfinite(probability)
        assert probability == pytest.approx(expected)


def test_unseen_valid_context_backs_off(vocab):
    """An unseen in-vocabulary context backs off to unigram."""

    model = NGramPrior.from_vocabularies(
        vocab,
        config=NGramConfig(
            order=2,
            alpha=0.1,
            streams=("chord",),
        ),
    )

    cmaj = state("Cmaj", "C")
    g7 = state("G7", "C")

    model.fit([[cmaj, g7]])

    seen = {
        context[0]
        for context in model.counts["chord"]
        if len(context) == 1
    }

    unseen = next(
        token
        for token in range(model.vocabulary_sizes["chord"])
        if token not in seen
    )

    probability = model.probability(
        "chord",
        g7.chord_id,
        history=(unseen,),
    )

    unigram_probability = model.probability(
        "chord",
        g7.chord_id,
        history=(),
    )

    assert math.isfinite(math.log(probability))
    assert probability == pytest.approx(unigram_probability)


def test_suggest_chords_uses_real_context_history(vocab):
    """The proposal method must use the supplied history."""

    earlier = state("Cmaj", "C")
    previous = state("Fmaj", "C")
    preferred = state("G7", "C")

    alternative_start = state("Dmaj", "C")
    alternative_next = state("Cmaj", "C")

    model = NGramPrior.from_vocabularies(
        vocab,
        config=NGramConfig(
            order=3,
            alpha=0.1,
            streams=("chord",),
        ),
    )

    model.fit(
        [[earlier, previous, preferred]] * 10
        + [[alternative_start, previous, alternative_next]] * 10
    )

    end_layer = Layer(
        time_index=2,
        states=(preferred,),
    )

    base_context = _build_prior_context(
        previous,
        end_layer,
        time_index=1,
    )

    context = replace(
    base_context,
    history=(earlier, previous),
    history_tokens=None,
    )

    proposals = model.suggest_chords(
        prev_state=previous,
        context=context,
        key_id=previous.key_id,
        k=1,
    )

    expected = model.top_k(
        "chord",
        (earlier.chord_id, previous.chord_id),
        k=1,
    )

    assert proposals == (expected[0][0],)
    assert proposals == (preferred.chord_id,)


def test_suggest_chords_identical_after_reload(tmp_path, vocab):
    """Save/load must preserve chord suggestions."""

    previous = state("Cmaj", "C")
    preferred = state("G7", "C")

    model = NGramPrior.from_vocabularies(
        vocab,
        config=NGramConfig(
            order=2,
            alpha=0.1,
            streams=("chord", "key"),
        ),
    )

    model.fit(
        [[previous, preferred, previous, preferred]] * 5
    )

    before = model.suggest_chords(
        prev_state=previous,
        context=None,
        key_id=previous.key_id,
        k=3,
    )

    path = tmp_path / "ngram.json"
    model.save(path)

    loaded = NGramPrior.load(path)

    after = loaded.suggest_chords(
        prev_state=previous,
        context=None,
        key_id=previous.key_id,
        k=3,
    )

    assert before == after
    assert preferred.chord_id in after
