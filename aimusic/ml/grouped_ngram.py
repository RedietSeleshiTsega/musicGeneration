
"""Joint key-chord N-gram prior.

Models (key, chord) as one combined token instead of
modeling the two streams independently.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from aimusic.core.core_types import BeatState
from aimusic.ml.dataset import vocab_sizes_from_vocabularies
from aimusic.ml.ngram_prior import NGramConfig, NGramPrior
from aimusic.scoring.priors import PriorContext
from aimusic.ml.ngram_vocab import (
    snapshot_vocabularies,
    validate_vocabulary_snapshot,
)


class GroupedNGramPrior:
    FORMAT_VERSION = 2

    def __init__(
        self,
        key_size: int,
        chord_size: int,
        order: int = 3,
        alpha: float = 0.1,
        provenance: str | dict = "",
        *,
        vocabulary_tokens: dict | None = None,
    ):
        if key_size < 1 or chord_size < 1:
            raise ValueError("Vocabulary sizes must be positive")

        self.key_size = key_size
        self.chord_size = chord_size
        self.order = order
        self.alpha = alpha
        self.provenance = provenance
        self.vocabulary_tokens = vocabulary_tokens

        # Reuse the existing N-gram probability engine.
        # Its internal "chord" stream stores joint token IDs.
        self.engine = NGramPrior(
            NGramConfig(
                order=order,
                alpha=alpha,
                streams=("chord",),
            ),
            {"chord": key_size * chord_size},
            provenance=provenance,
        )

    @classmethod
    def from_vocabularies(
        cls,
        vocabularies,
        order=3,
        alpha=0.1,
        provenance="",
    ):
        sizes = vocab_sizes_from_vocabularies(vocabularies)

        return cls(
            sizes["key"],
            sizes["chord"],
            order,
            alpha,
            provenance,
            vocabulary_tokens=snapshot_vocabularies(
                vocabularies,
                ("key", "chord"),
            ),
        )

    def encode(self, state: BeatState) -> int:
        """Convert a key-chord pair to one token ID."""
        key = int(state.key_id)
        chord = int(state.chord_id)

        if not (0 <= key < self.key_size):
            return self.key_size * self.chord_size

        if not (0 <= chord < self.chord_size):
            return self.key_size * self.chord_size

        return key * self.chord_size + chord

    def decode(self, token: int) -> tuple[int, int]:
        """Convert a joint token back into key and chord IDs."""
        if not 0 <= token < self.key_size * self.chord_size:
            raise ValueError("Unknown joint token")

        return divmod(token, self.chord_size)

    def fit(self, pieces):
        """Train without crossing song boundaries."""
        self.engine.counts = {
            "chord": defaultdict(Counter)
        }

        table = self.engine.counts["chord"]

        for piece in pieces:
            tokens = [self.encode(state) for state in piece]

            for i, token in enumerate(tokens):
                max_history = min(i, self.order - 1)

                for length in range(max_history + 1):
                    context = tuple(tokens[i-length:i])
                    table[context][token] += 1

        return self

    def probability(self, next_state, history=()):
        return self.engine.probability(
            "chord",
            self.encode(next_state),
            tuple(self.encode(state) for state in history),
        )

    def logp_next(
        self,
        prev_state: BeatState,
        next_state: BeatState,
        t: int,
        context: PriorContext | None = None,
    ) -> float:
        history = list(context.history) if context else []

        # PriorContext.history usually already contains prev_state.
        # Add it only when the previous state is absent.
        if not history or history[-1] != prev_state:
            history.append(prev_state)

        return math.log(
            self.probability(next_state, history)
        )

    def logp_next_batch(self, queries):
        return tuple(
            self.logp_next(
                query.prev_state,
                query.next_state,
                query.time_index,
                query.context,
            )
            for query in queries
        )

    def top_k(self, history=(), k=5):
        """Suggest likely next (key_id, chord_id) combinations."""
        if k < 1:
            raise ValueError("k must be positive")

        ids = tuple(self.encode(state) for state in history)

        results = self.engine.top_k(
            "chord",
            ids,
            k=k,
        )

        return tuple(
            (self.decode(token), probability)
            for token, probability in results
        )

    
    def suggest_chords(
        self,
        prev_state: BeatState,
        context: PriorContext | None,
        key_id: int,
        k: int = 3,
    ) -> tuple[int, ...]:
        """Suggest chord IDs from the learned joint distribution."""

        if k < 1:
            raise ValueError("k must be positive")

        history = list(context.history) if context else []

        if not history or history[-1] != prev_state:
            history.append(prev_state)

        # Examine all known joint key-chord combinations.
        suggestions = self.top_k(
            history=tuple(history),
            k=self.key_size * self.chord_size,
        )

        result = []

        for (suggested_key, chord_id), probability in suggestions:
            if suggested_key != key_id:
                continue

            if chord_id not in result:
                result.append(chord_id)

            if len(result) >= k:
                break

        return tuple(result)


    def save(self, path):
        """Save counts and configuration in one JSON file."""
        payload = {
            "version": (
                self.FORMAT_VERSION
                if self.vocabulary_tokens is not None
                else 1
            ),
            "vocabulary_tokens": self.vocabulary_tokens,
            "model": "grouped_key_chord_ngram",
            "key_size": self.key_size,
            "chord_size": self.chord_size,
            "order": self.order,
            "alpha": self.alpha,
            "provenance": self.provenance,
            "counts": [
                {
                    "context": list(context),
                    "next": [
                        [token, count]
                        for token, count in sorted(row.items())
                    ],
                }
                for context, row in sorted(
                    self.engine.counts["chord"].items()
                )
            ],
        }

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def validate_vocabularies(self, vocabularies):
        validate_vocabulary_snapshot(
            self.vocabulary_tokens,
            vocabularies,
        )

    @classmethod
    def load(cls, path):
        payload = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        if payload["version"] not in (1, cls.FORMAT_VERSION):
            raise ValueError("Unsupported grouped model version")

        model = cls(
            payload["key_size"],
            payload["chord_size"],
            payload["order"],
            payload["alpha"],
            payload.get("provenance", ""),
            vocabulary_tokens=payload.get("vocabulary_tokens"),
        )

        table = model.engine.counts["chord"]

        for row in payload["counts"]:
            context = tuple(row["context"])

            table[context] = Counter({
                int(token): int(count)
                for token, count in row["next"]
            })

        return model
