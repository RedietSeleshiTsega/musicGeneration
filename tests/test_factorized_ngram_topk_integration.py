
"""Verify factorized NGramPrior proposals reach candidate generation."""

from aimusic.core.config import StyleConfig
from aimusic.core.vocab import build_default_vocabularies
from aimusic.ml.ngram_prior import NGramConfig, NGramPrior
from aimusic.planning.candidates import _top_k_prior_chord_ids
from tests.test_grouped_ngram import state


def test_factorized_ngram_topk_is_consumed(monkeypatch):
    vocab = build_default_vocabularies(StyleConfig())

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

    calls = []
    original_top_k = model.top_k

    def tracked_top_k(stream, history, k=5):
        calls.append((stream, history, k))
        return original_top_k(stream, history, k=k)

    monkeypatch.setattr(model, "top_k", tracked_top_k)

    proposals = _top_k_prior_chord_ids(
        previous,
        previous.meter_id,
        1,
        0,
        previous.key_id,
        previous.role_id,
        previous.groove_id,
        prior=model,
        context=None,
        vocabularies=vocab,
        top_k=3,
    )

    assert len(calls) == 1
    assert calls[0][0] == "chord"
    assert calls[0][1] == (previous.chord_id,)

    assert preferred.chord_id in proposals
    assert len(proposals) <= 3
