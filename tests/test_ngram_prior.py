
import math

import pytest

from aimusic.core.core_types import BeatState
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.ml.dataset import build_transition_examples
from aimusic.ml.ngram_prior import NGramConfig, NGramPrior
from aimusic.scoring.priors import PriorContext


def state(chord="Cmaj", key="C"):
    v = DEFAULT_VOCABULARIES
    return BeatState(
        meter_id=v.meters.token_for_label("4/4").id,
        beat_in_bar=0,
        boundary_lvl=0,
        key_id=v.keys.token_for_label(key).id,
        chord_id=v.chords.token_for_label(chord).id,
        role_id=v.roles.token_for_label("hold").id,
        head_id=v.heads.token_for_label("root").id,
        groove_id=v.grooves.token_for_label("straight_8ths").id,
    )


def trained(order=3, streams=("chord",)):
    model = NGramPrior.from_vocabularies(
        DEFAULT_VOCABULARIES,
        config=NGramConfig(
            order=order,
            alpha=0.1,
            streams=streams,
        ),
    )

    sequence = (
        state("Cmaj"),
        state("G7"),
        state("Amin"),
        state("Cmaj"),
        state("G7"),
        state("Amin"),
    )

    model.fit([sequence] * 3)
    return model


def test_frequent_transition():
    model = trained(order=2)

    good = model.logp_next(
        state("Cmaj"), state("G7"), 0
    )

    rare = model.logp_next(
        state("Cmaj"), state("Dmaj"), 0
    )

    assert good > rare


def test_trigram_context():
    model = trained(order=3)

    context = PriorContext(
        history=(state("Cmaj"), state("G7"))
    )

    good = model.logp_next(
        state("G7"), state("Amin"), 1, context
    )

    rare = model.logp_next(
        state("G7"), state("Dmaj"), 1, context
    )

    assert good > rare


def test_unseen_context():
    model = trained()

    assert math.isfinite(
        model.probability("chord", 999, (1000,))
    )


def test_save_reload_and_batch(tmp_path):
    model = trained(
        streams=("chord", "key")
    )

    queries = build_transition_examples(
        [state("Cmaj"), state("G7"), state("Amin")],
        history_len=2,
    )

    original = model.logp_next_batch(queries)

    path = tmp_path / "ngram.json"
    model.save(path)

    reloaded = NGramPrior.load(path)

    assert original == reloaded.logp_next_batch(queries)


def test_top_k():
    model = trained()

    chord_id = state("Cmaj").chord_id

    suggestions = model.top_k(
        "chord", (chord_id,), k=5
    )

    assert len(suggestions) == 5
    assert suggestions[0][0] == state("G7").chord_id


def test_partial_model():
    model = trained(streams=("chord",))

    score1 = model.logp_next(
        state("Cmaj", "C"),
        state("G7", "C"),
        0,
    )

    score2 = model.logp_next(
        state("Cmaj", "G"),
        state("G7", "G"),
        0,
    )

    assert score1 == score2
