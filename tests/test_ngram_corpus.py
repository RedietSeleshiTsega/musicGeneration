
"""Tests for canonical corpus records."""

import hashlib

import pytest

from aimusic.ml.ngram_corpus import (
    NGramCorpusRecord,
    corpus_provenance,
)
from tests.test_grouped_ngram import state


def make_record(piece_id, states):
    return NGramCorpusRecord(
        piece_id=piece_id,
        source_path=f"/example/{piece_id}.mid",
        source_sha256=hashlib.sha256(
            piece_id.encode()
        ).hexdigest(),
        annotation_method="heuristic_midi_v1",
        states=tuple(states),
    )


def test_record_preserves_song_identity():
    record = make_record(
        "song_001",
        [state("Cmaj"), state("G7")],
    )

    assert record.piece_id == "song_001"
    assert len(record.states) == 2


def test_record_tokenizes_beatstates():
    record = make_record(
        "song_001",
        [state("Cmaj"), state("G7")],
    )

    tokens = record.structural_tokens()

    assert len(tokens) == 2
    assert tokens[0].chord_id == record.states[0].chord_id
    assert tokens[1].chord_id == record.states[1].chord_id


def test_empty_record_rejected():
    with pytest.raises(ValueError):
        make_record("empty", [])


def test_provenance_includes_every_piece():
    first = make_record("001", [state("Cmaj")])
    second = make_record("002", [state("G7")])

    metadata = corpus_provenance((first, second))

    assert metadata["piece_count"] == 2
    assert len(metadata["pieces"]) == 2
    assert metadata["pieces"][0]["piece_id"] == "001"
    assert metadata["pieces"][1]["piece_id"] == "002"
