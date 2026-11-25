
"""Verify factorized NGramPrior proposals reach candidate generation."""

from aimusic.core.config import StyleConfig
from aimusic.core.core_types import BeatState
from aimusic.core.vocab import build_default_vocabularies
from aimusic.ml.ngram_prior import NGramConfig, NGramPrior
from aimusic.planning.candidates import _top_k_prior_chord_ids


def make_state(chord_label, key_label, vocab):
    """Build a BeatState using the project's actual token IDs."""
    return BeatState(
        meter_id=0,
        beat_in_bar=0,
        boundary_lvl=0,
        key_id=vocab.keys.token_for_label(key_label).id,
        chord_id=vocab.chords.token_for_label(chord_label).id,
        role_id=0,
        head_id=1,
        groove_id=0,
    )


def test_factorized_ngram_topk_is_consumed(monkeypatch):
    vocab = build_default_vocabularies(StyleConfig())

    previous = make_state("Cmaj", "C", vocab)
    preferred = make_state("G7", "C", vocab)

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
