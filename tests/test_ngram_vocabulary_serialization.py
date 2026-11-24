
"""Tests for versioned N-gram vocabulary serialization."""

import pytest

from aimusic.core.config import StyleConfig
from aimusic.core.vocab import build_default_vocabularies
from aimusic.ml.ngram_prior import NGramConfig, NGramPrior


@pytest.fixture
def vocab():
    return build_default_vocabularies(StyleConfig())


def test_individual_vocab_roundtrip(tmp_path, vocab):
    model = NGramPrior.from_vocabularies(
        vocab,
        config=NGramConfig(
            order=2,
            streams=("chord", "key"),
        ),
    )

    path = tmp_path / "individual.json"
    model.save(path)

    loaded = NGramPrior.load(path)

    assert loaded.vocabulary_tokens == model.vocabulary_tokens
    assert loaded.vocabulary_tokens["chord"]
    assert loaded.vocabulary_tokens["key"]

    loaded.validate_vocabularies(vocab)


def test_individual_vocab_mismatch_detected(tmp_path, vocab):
    model = NGramPrior.from_vocabularies(
        vocab,
        config=NGramConfig(streams=("chord",)),
    )

    path = tmp_path / "model.json"
    model.save(path)

    loaded = NGramPrior.load(path)

    # Simulate a model with a different ID-to-label mapping.
    loaded.vocabulary_tokens["chord"][0]["label"] = (
        "incorrect_chord"
    )

    with pytest.raises(ValueError, match="Vocabulary mismatch"):
        loaded.validate_vocabularies(vocab)


def test_legacy_individual_model_loads(tmp_path, vocab):
    model = NGramPrior.from_vocabularies(
        vocab,
        config=NGramConfig(streams=("chord",)),
    )

    model.vocabulary_tokens = None

    path = tmp_path / "legacy.json"
    model.save(path)

    loaded = NGramPrior.load(path)

    assert loaded.vocabulary_tokens is None


