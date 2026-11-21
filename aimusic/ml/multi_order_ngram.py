
"""Adaptive interpolation of grouped unigram, bigram and trigram models."""

from __future__ import annotations

import math

from aimusic.ml.grouped_ngram import GroupedNGramPrior


class MultiOrderNGramPrior:
    def __init__(
        self,
        unigram: GroupedNGramPrior,
        bigram: GroupedNGramPrior,
        trigram: GroupedNGramPrior,
        strength: float = 20.0,
    ):
        models = (unigram, bigram, trigram)

        if tuple(model.order for model in models) != (1, 2, 3):
            raise ValueError("Expected models of orders 1, 2 and 3")

        if len({(m.key_size, m.chord_size) for m in models}) != 1:
            raise ValueError("Model vocabulary sizes must match")

        if not math.isfinite(strength) or strength <= 0:
            raise ValueError("strength must be positive and finite")

        self.unigram = unigram
        self.bigram = bigram
        self.trigram = trigram
        self.strength = strength

    def _history(self, prev_state, context):
        history = list(context.history) if context else []

        if not history or history[-1] != prev_state:
            history.append(prev_state)

        return history

    def _weight(self, model, history):
        length = model.order - 1
        tokens = tuple(
            model.encode(state) for state in history[-length:]
        ) if length else ()

        # Require the complete context for this N-gram order.
        if len(tokens) < length:
            return 0.0

        row = model.engine.counts["chord"].get(tokens)
        count = sum(row.values()) if row else 0

        return count / (count + self.strength)

    def logp_next(self, prev_state, next_state, t, context=None):
        history = self._history(prev_state, context)

        p1 = math.exp(self.unigram.logp_next(
            prev_state, next_state, t, context
        ))
        p2 = math.exp(self.bigram.logp_next(
            prev_state, next_state, t, context
        ))
        p3 = math.exp(self.trigram.logp_next(
            prev_state, next_state, t, context
        ))

        w2 = self._weight(self.bigram, history)
        w3 = self._weight(self.trigram, history)

        p12 = w2 * p2 + (1.0 - w2) * p1
        probability = w3 * p3 + (1.0 - w3) * p12

        return math.log(probability)

    def logp_next_batch(self, queries):
        return tuple(
            self.logp_next(
                q.prev_state,
                q.next_state,
                q.time_index,
                q.context,
            )
            for q in queries
        )
