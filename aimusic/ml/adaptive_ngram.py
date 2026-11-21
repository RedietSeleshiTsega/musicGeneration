
"""Adaptive interpolation of individual and grouped N-gram priors."""

from __future__ import annotations

import math

from aimusic.ml.ngram_prior import NGramPrior
from aimusic.ml.grouped_ngram import GroupedNGramPrior
from aimusic.scoring.priors import PriorContext


class AdaptiveNGramPrior:
    def __init__(
        self,
        individual: NGramPrior,
        grouped: GroupedNGramPrior,
        strength: float = 20.0,
    ):
        if not math.isfinite(strength) or strength <= 0:
            raise ValueError("strength must be positive and finite")

        if set(individual.config.streams) != {"chord", "key"}:
            raise ValueError(
                "Individual model must use chord and key streams"
            )

        if (
            individual.vocabulary_sizes["chord"]
            != grouped.chord_size
            or individual.vocabulary_sizes["key"]
            != grouped.key_size
        ):
            raise ValueError("Model vocabularies do not match")

        self.individual = individual
        self.grouped = grouped
        self.strength = strength

    def _history(self, prev_state, context):
        history = list(context.history) if context else []

        if not history or history[-1] != prev_state:
            history.append(prev_state)

        return history

    def grouped_weight(self, prev_state, context=None):
        """Estimate how reliable the grouped context is."""

        history = self._history(prev_state, context)

        token_history = tuple(
            self.grouped.encode(state) for state in history
        )

        max_length = self.grouped.order - 1

        if max_length == 0:
            token_history = ()
        else:
            token_history = token_history[-max_length:]

        # Only trust the exact requested context here.
        # If it is unseen, prefer the individual model.
        row = self.grouped.engine.counts["chord"].get(
            token_history
        )

        count = sum(row.values()) if row else 0

        return count / (count + self.strength)

    def logp_next(
        self,
        prev_state,
        next_state,
        t,
        context: PriorContext | None = None,
    ):
        individual_score = self.individual.logp_next(
            prev_state, next_state, t, context
        )

        grouped_score = self.grouped.logp_next(
            prev_state, next_state, t, context
        )

        weight = self.grouped_weight(
            prev_state, context
        )

        # Interpolate probabilities, NOT log scores.
        probability = (
            (1.0 - weight) * math.exp(individual_score)
            + weight * math.exp(grouped_score)
        )

        return math.log(probability)

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
