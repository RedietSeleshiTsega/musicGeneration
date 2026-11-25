
"""Factorized corpus-backed N-gram prior."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from aimusic.core.core_types import BeatState
from aimusic.core.vocab import Vocabularies
from aimusic.ml.dataset import (
    STREAM_FIELD_MAP,
    vocab_sizes_from_vocabularies,
)
from aimusic.scoring.priors import (
    PriorContext,
    PriorQuery,
    STRUCTURAL_STREAM_NAMES,
    StructuralEventTokens,
    TokenizedPriorQuery,
)
from aimusic.ml.ngram_vocab import (
    snapshot_vocabularies,
    validate_vocabulary_snapshot,
)


@dataclass(frozen=True)
class NGramConfig:
    order: int = 3
    alpha: float = 0.1
    streams: tuple[str, ...] = STRUCTURAL_STREAM_NAMES

    def __post_init__(self):
        if (
            not isinstance(self.order, int)
            or isinstance(self.order, bool)
            or self.order < 1
        ):
            raise ValueError("order must be a positive integer")

        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be positive and finite")

        if not self.streams or len(set(self.streams)) != len(self.streams):
            raise ValueError("streams must be nonempty and unique")

        if any(s not in STRUCTURAL_STREAM_NAMES for s in self.streams):
            raise ValueError("Unknown structural stream")


class NGramPrior:
    """Train N-grams independently for selected musical attributes."""

    FORMAT_VERSION = 2

    def __init__(
        self,
        config: NGramConfig,
        vocabulary_sizes: Mapping[str, int],
        *,
        provenance: str | dict = "",
        vocabulary_tokens: dict | None = None,
    ):
        self.config = config

        self.vocabulary_sizes = {
            s: int(vocabulary_sizes[s])
            for s in config.streams
        }

        if any(size <= 0 for size in self.vocabulary_sizes.values()):
            raise ValueError("Vocabulary sizes must be positive")

        self.provenance = provenance
        self.vocabulary_tokens = vocabulary_tokens

        self.counts: dict[
            str,
            defaultdict[tuple[int, ...], Counter[int]]
        ] = {
            s: defaultdict(Counter)
            for s in config.streams
        }

    @classmethod
    def from_vocabularies(
        cls,
        vocabularies: Vocabularies,
        *,
        config: NGramConfig | None = None,
        provenance: str | dict = "",
    ):
        selected_config = config or NGramConfig()

        return cls(
            selected_config,
            vocab_sizes_from_vocabularies(vocabularies),
            provenance=provenance,
            vocabulary_tokens=snapshot_vocabularies(
                vocabularies,
                selected_config.streams,
            ),
        )

    def _normalize(self, stream: str, token: int) -> int:
        size = self.vocabulary_sizes[stream]

        if 0 <= token < size:
            return token

        return size  # Unknown-token bucket

    def fit(
        self,
        pieces: Sequence[Sequence[BeatState]],
    ):
        """Learn counts without crossing musical-piece boundaries."""

        self.counts = {
            s: defaultdict(Counter)
            for s in self.config.streams
        }

        for piece in pieces:
            states = tuple(piece)

            if any(not isinstance(st, BeatState) for st in states):
                raise TypeError("Expected BeatState objects")

            for stream in self.config.streams:
                field = STREAM_FIELD_MAP[stream][0]

                tokens = tuple(
                    self._normalize(
                        stream,
                        int(
                            getattr(
                                StructuralEventTokens.from_state(st),
                                field,
                            )
                        ),
                    )
                    for st in states
                )

                for i, token in enumerate(tokens):
                    max_context = min(
                        i,
                        self.config.order - 1,
                    )

                    for size in range(max_context + 1):
                        context = tokens[i-size:i]
                        self.counts[stream][context][token] += 1

        return self

    def probability(
        self,
        stream: str,
        token: int,
        history: Sequence[int] = (),
    ) -> float:
        """Smoothed probability with context backoff."""

        if stream not in self.counts:
            raise ValueError(f"Inactive stream: {stream}")

        token = self._normalize(stream, int(token))

        context = tuple(
            self._normalize(stream, int(t))
            for t in history
        )

        if self.config.order > 1:
            context = context[-(self.config.order - 1):]
        else:
            context = ()

        counts = self.counts[stream]

        # Back off until an observed context is found.
        while context and not counts.get(context):
            context = context[1:]

        row = counts.get(context, Counter())
        total = sum(row.values())

        vocab_size = self.vocabulary_sizes[stream] + 1

        return (
            row[token] + self.config.alpha
        ) / (
            total + self.config.alpha * vocab_size
        )

    def _history(
        self,
        stream: str,
        query: TokenizedPriorQuery,
    ) -> tuple[int, ...]:

        field = STREAM_FIELD_MAP[stream][0]

        sequence_fields = {
            "meter_id": "meter_ids",
            "beat_position": "beat_positions",
            "boundary_level": "boundary_levels",
            "key_id": "key_ids",
            "chord_id": "chord_ids",
            "role_id": "role_ids",
            "head_id": "head_ids",
            "groove_id": "groove_ids",
        }

        history = tuple(
            getattr(
                query.history_tokens,
                sequence_fields[field],
            )
        )

        previous = int(
            getattr(query.prev_event, field)
        )

        if not history or history[-1] != previous:
            history += (previous,)

        return history

    def score_transition(
        self,
        query: TokenizedPriorQuery,
    ) -> float:
        """Sum the log probabilities of active streams."""

        total = 0.0

        for stream in self.config.streams:
            field = STREAM_FIELD_MAP[stream][1]

            next_token = int(
                getattr(query.next_event, field)
            )

            history = self._history(stream, query)

            probability = self.probability(
                stream,
                next_token,
                history,
            )

            total += math.log(probability)

        return total

    def score_transition_batch(
        self,
        queries: Sequence[TokenizedPriorQuery],
    ) -> tuple[float, ...]:
        return tuple(
            self.score_transition(q)
            for q in queries
        )

    def logp_next(
        self,
        prev_state: BeatState,
        next_state: BeatState,
        t: int,
        context: PriorContext | None = None,
    ) -> float:

        query = PriorQuery(
            prev_state,
            next_state,
            t,
            context,
        ).tokenize()

        return self.score_transition(query)

    def logp_next_batch(
        self,
        queries: Sequence[PriorQuery],
    ) -> tuple[float, ...]:

        return tuple(
            self.logp_next(
                q.prev_state,
                q.next_state,
                q.time_index,
                q.context,
            )
            for q in queries
        )

    def top_k(
        self,
        stream: str,
        history: Sequence[int],
        k: int = 5,
    ) -> tuple[tuple[int, float], ...]:
        """Return likely next tokens and probabilities."""

        if k < 1:
            raise ValueError("k must be positive")

        size = self.vocabulary_sizes[stream]

        candidates = (
            (
                token,
                self.probability(stream, token, history),
            )
            for token in range(size)
        )

        return tuple(
            sorted(
                candidates,
                key=lambda item: (-item[1], item[0]),
            )[:k]
        )

    
    def suggest_chords(
        self,
        prev_state: BeatState,
        context: PriorContext | None,
        key_id: int,
        k: int = 3,
    ) -> tuple[int, ...]:
        """Suggest chord IDs using the factorized chord stream.

        The key_id is accepted to satisfy the shared proposal
        interface. Chord probabilities are modeled independently
        of the key stream in this factorized prior.
        """
        del key_id

        if k < 1:
            raise ValueError("k must be positive")

        if "chord" not in self.config.streams:
            return ()

        history = list(context.history) if context else []

        if not history or history[-1] != prev_state:
            history.append(prev_state)

        chord_history = tuple(
            int(state.chord_id)
            for state in history
        )

        suggestions = self.top_k(
            "chord",
            chord_history,
            k=k,
        )

        return tuple(
            chord_id
            for chord_id, probability in suggestions
        )


    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "version": (
                self.FORMAT_VERSION
                if self.vocabulary_tokens is not None
                else 1
            ),
            "vocabulary_tokens": self.vocabulary_tokens,
            "order": self.config.order,
            "alpha": self.config.alpha,
            "streams": list(self.config.streams),
            "vocabulary_sizes": self.vocabulary_sizes,
            "provenance": self.provenance,
            "counts": {
                stream: [
                    {
                        "context": list(context),
                        "next": [
                            [token, count]
                            for token, count in sorted(row.items())
                        ],
                    }
                    for context, row in sorted(
                        self.counts[stream].items()
                    )
                ]
                for stream in self.config.streams
            },
        }

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
    def load(cls, path: str | Path):
        payload = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        if payload["version"] not in (1, cls.FORMAT_VERSION):
            raise ValueError("Unsupported NGramPrior format")

        config = NGramConfig(
            order=payload["order"],
            alpha=payload["alpha"],
            streams=tuple(payload["streams"]),
        )

        model = cls(
            config,
            payload["vocabulary_sizes"],
            provenance=payload.get("provenance", ""),
            vocabulary_tokens=payload.get("vocabulary_tokens"),
        )

        for stream in config.streams:
            for item in payload["counts"][stream]:
                context = tuple(item["context"])

                model.counts[stream][context] = Counter({
                    int(token): int(count)
                    for token, count in item["next"]
                })

        return model
