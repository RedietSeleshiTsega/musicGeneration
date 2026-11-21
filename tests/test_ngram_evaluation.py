"""Tests for the held-out baseline evaluation report."""

import json
import math

import pytest

from aimusic.core.config import StyleConfig
from aimusic.core.vocab import build_default_vocabularies
from aimusic.ml.ngram_corpus import NGramCorpusRecord
from aimusic.ml.ngram_evaluation import evaluate_heldout, save_report
from aimusic.ml.ngram_prior import NGramConfig, NGramPrior
from tests.test_grouped_ngram import state


def record(piece_id, states):
    return NGramCorpusRecord(
        piece_id=piece_id,
        source_path=f"{piece_id}.mid",
        source_sha256="test_hash",
        annotation_method="test",
        states=tuple(states),
    )


def test_evaluation_baselines_and_json(tmp_path):
    vocab = build_default_vocabularies(StyleConfig())
    training = [record("train", [state("Cmaj"), state("G7")] * 5)]
    heldout = [record("test", [state("Cmaj"), state("G7"), state("Cmaj")])]
    model = NGramPrior.from_vocabularies(
        vocab,
        config=NGramConfig(order=2, streams=("chord", "key"), alpha=0.1),
    )
    model.fit([r.states for r in training])

    report = evaluate_heldout(model, training, heldout, vocab)
    assert report["heldout_transitions"] == 2
    for name in ("ngram", "unigram", "null_prior_uniform_surrogate"):
        metrics = report["models"][name]
        assert math.isfinite(metrics["nll"])
        assert math.isfinite(metrics["perplexity"])
        assert metrics["perplexity"] == pytest.approx(math.exp(metrics["nll"]))

    expected_uniform_nll = math.log(49) + math.log(13)
    assert report["models"]["null_prior_uniform_surrogate"]["nll"] == pytest.approx(expected_uniform_nll)
    assert report["models"]["null_prior_raw"]["nll"] is None
    assert report["models"]["null_prior_raw"]["perplexity"] is None

    path = save_report(report, tmp_path / "heldout.json")
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_evaluation_rejects_leakage():
    vocab = build_default_vocabularies(StyleConfig())
    record_a = record("same", [state("Cmaj"), state("G7")])
    model = NGramPrior.from_vocabularies(
        vocab, config=NGramConfig(streams=("chord",)),
    )
    model.fit([record_a.states])
    with pytest.raises(ValueError, match="overlap"):
        evaluate_heldout(model, [record_a], [record_a], vocab)
