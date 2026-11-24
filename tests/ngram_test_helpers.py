
"""Shared BeatState helpers for N-gram tests."""

from aimusic.core.config import StyleConfig
from aimusic.core.core_types import BeatState
from aimusic.core.vocab import build_default_vocabularies


def state(chord_label: str, key_label: str = "C") -> BeatState:
    """Create a valid BeatState from chord and key labels."""
    vocab = build_default_vocabularies(StyleConfig())

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
