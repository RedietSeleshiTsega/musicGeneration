
"""Test direct NGramPrior suggestions in candidate generation."""

from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.ml.grouped_ngram import GroupedNGramPrior
from aimusic.planning.candidates import _top_k_prior_chord_ids
from tests.test_grouped_ngram import state


def test_grouped_ngram_topk_is_consumed(monkeypatch):
    vocab = DEFAULT_VOCABULARIES

    previous = state("Cmaj", "C")
    preferred = state("G7", "C")

    model = GroupedNGramPrior.from_vocabularies(
        vocab,
        order=2,
        alpha=0.1,
    )

    model.fit([
        [previous, preferred, previous, preferred]
    ] * 5)

    calls = []
    original_top_k = model.top_k

    def tracked_top_k(history=(), k=5):
        calls.append((history, k))
        return original_top_k(history=history, k=k)

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

    # Verify our NGramPrior's own top_k method was called.
    assert len(calls) == 1
    assert calls[0][0] == (previous,)

    # Verify the learned preferred chord is proposed.
    assert preferred.chord_id in proposals
    assert len(proposals) <= 3

    # Verify every proposed chord exists in the vocabulary.
    assert all(vocab.chords.has_id(chord) for chord in proposals)
