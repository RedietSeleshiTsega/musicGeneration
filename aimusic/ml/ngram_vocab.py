
"""Vocabulary snapshots for serialized N-gram models."""

from aimusic.core.vocab import Vocabularies


STREAM_VOCABULARY_ATTRIBUTES = {
    "meter": "meters",
    "beat_position": "beat_positions",
    "boundary": "boundaries",
    "key": "keys",
    "chord": "chords",
    "role": "roles",
    "head": "heads",
    "groove": "grooves",
}


def snapshot_vocabularies(
    vocabularies: Vocabularies,
    streams,
) -> dict:
    """Save the exact ID-to-label mapping of selected streams."""

    result = {}

    for stream in streams:
        attribute = STREAM_VOCABULARY_ATTRIBUTES[stream]
        vocabulary = getattr(vocabularies, attribute)

        tokens = sorted(vocabulary, key=lambda token: token.id)

        # NGramPrior indexes tokens using consecutive integers.
        ids = [token.id for token in tokens]
        if ids != list(range(len(tokens))):
            raise ValueError(
                f"{stream} IDs must be consecutive from zero"
            )

        result[stream] = [
            {
                "id": token.id,
                "label": token.label,
            }
            for token in tokens
        ]

    return result


def validate_vocabulary_snapshot(
    snapshot: dict,
    vocabularies: Vocabularies,
) -> None:
    """Reject incompatible vocabularies before scoring."""

    if snapshot is None:
        raise ValueError(
            "Legacy model has no vocabulary snapshot. "
            "Retrain to enable vocabulary validation."
        )

    actual = snapshot_vocabularies(
        vocabularies,
        snapshot.keys(),
    )

    if actual != snapshot:
        raise ValueError(
            "Vocabulary mismatch: model token IDs or labels "
            "do not match the supplied vocabularies."
        )
