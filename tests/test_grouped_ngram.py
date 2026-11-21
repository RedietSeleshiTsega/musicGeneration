
import math

import pytest

from aimusic.core.core_types import BeatState
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.ml.dataset import build_transition_examples
from aimusic.ml.grouped_ngram import GroupedNGramPrior


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


def trained():
    model = GroupedNGramPrior.from_vocabularies(
        DEFAULT_VOCABULARIES,
        order=3,
        alpha=0.1,
    )

    piece = [
        state("Cmaj", "C"),
        state("G7", "C"),
        state("Amin", "C"),
        state("Cmaj", "C"),
        state("G7", "C"),
        state("Amin", "C"),
    ]

    model.fit([piece] * 3)
    return model


def test_joint_encoding():
    model = trained()
    example = state("G7", "C")

    key, chord = model.decode(model.encode(example))

    assert key == example.key_id
    assert chord == example.chord_id


def test_joint_transition():
    model = trained()

    previous = state("Cmaj", "C")

    common = model.logp_next(
        previous,
        state("G7", "C"),
        0,
    )

    rare = model.logp_next(
        previous,
        state("G7", "G"),
        0,
    )

    assert common > rare


def test_unseen_context():
    model = trained()

    score = model.logp_next(
        state("Dmaj", "D"),
        state("Cmaj", "C"),
        0,
    )

    assert math.isfinite(score)


def test_save_reload(tmp_path):
    model = trained()

    queries = build_transition_examples(
        [
            state("Cmaj", "C"),
            state("G7", "C"),
            state("Amin", "C"),
        ],
        history_len=2,
    )

    original = model.logp_next_batch(queries)

    path = tmp_path / "grouped.json"
    model.save(path)

    restored = GroupedNGramPrior.load(path)

    assert restored.logp_next_batch(queries) == original


def test_top_k():
    model = trained()

    suggestions = model.top_k(
        history=[state("Cmaj", "C")],
        k=5,
    )

    assert len(suggestions) == 5

    expected = (
        state("G7", "C").key_id,
        state("G7", "C").chord_id,
    )

    assert suggestions[0][0] == expected
