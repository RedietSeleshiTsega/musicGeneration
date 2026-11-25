
"""Canonical corpus records for corpus-backed N-gram training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from aimusic.core.core_types import BeatState
from aimusic.core.vocab import Vocabularies
from aimusic.ml.annotate import annotate_beat_grid
from aimusic.ml.midi_ingest import parse_midi_beats
from aimusic.scoring.priors import StructuralEventTokens


ANNOTATION_METHOD = "heuristic_midi_v1"


@dataclass(frozen=True)
class NGramCorpusRecord:
    """One song and its ordered musical structural states."""

    piece_id: str
    source_path: str
    source_sha256: str
    annotation_method: str
    states: tuple[BeatState, ...]

    def __post_init__(self):
        if not self.piece_id:
            raise ValueError("piece_id cannot be empty")

        if not self.states:
            raise ValueError("Corpus record requires at least one BeatState")

        if not all(isinstance(s, BeatState) for s in self.states):
            raise TypeError("states must contain BeatState objects")

    def structural_tokens(self) -> tuple[StructuralEventTokens, ...]:
        """Use the repository's canonical BeatState tokenization."""
        return tuple(
            StructuralEventTokens.from_state(state)
            for state in self.states
        )


def record_from_midi(
    path: str | Path,
    vocabularies: Vocabularies,
    *,
    corpus_root: str | Path | None = None,
    edo: int = 12,
) -> NGramCorpusRecord:
    """Convert one MIDI song to a provenance-aware corpus record."""

    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    if corpus_root is None:
        piece_id = path.name
    else:
        piece_id = path.relative_to(
            Path(corpus_root).resolve()
        ).as_posix()

    states = tuple(
        annotate_beat_grid(
            parse_midi_beats(path),
            vocabularies,
            edo=edo,
        )
    )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    return NGramCorpusRecord(
        piece_id=piece_id,
        source_path=str(path),
        source_sha256=digest,
        annotation_method=ANNOTATION_METHOD,
        states=states,
    )


def load_ngram_corpus(
    midi_dir: str | Path,
    vocabularies: Vocabularies,
    *,
    edo: int = 12,
) -> tuple[NGramCorpusRecord, ...]:
    """Load songs in deterministic order without mixing their histories."""

    root = Path(midi_dir).resolve()

    if not root.is_dir():
        raise NotADirectoryError(root)

    files = sorted(
        [
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in (".mid", ".midi")
        ],
        key=lambda p: p.relative_to(root).as_posix(),
    )

    records = []

    for path in files:
        record = record_from_midi(
            path,
            vocabularies,
            corpus_root=root,
            edo=edo,
        )
        records.append(record)

    if not records:
        raise ValueError("No usable MIDI records found")

    ids = [record.piece_id for record in records]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate piece IDs found")

    return tuple(records)


def corpus_provenance(records) -> dict:
    """Summarize the exact training sources."""

    records = tuple(records)

    if not records:
        raise ValueError("Cannot describe an empty corpus")

    return {
        "annotation_method": ANNOTATION_METHOD,
        "piece_count": len(records),
        "pieces": [
            {
                "piece_id": record.piece_id,
                "sha256": record.source_sha256,
                "beat_count": len(record.states),
            }
            for record in records
        ],
    }
