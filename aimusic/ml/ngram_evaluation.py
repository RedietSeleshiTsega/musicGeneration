"""Held-out evaluation against unigram and normalized uniform baselines."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

from aimusic.core.vocab import Vocabularies
from aimusic.ml.dataset import build_transition_examples
from aimusic.ml.ngram_corpus import NGramCorpusRecord
from aimusic.ml.ngram_prior import NGramConfig, NGramPrior
from aimusic.scoring.priors import NullPrior


def _metrics(log_scores: Sequence[float]) -> dict:
    """Return mean NLL and perplexity for normalized log probabilities."""
    if not log_scores:
        raise ValueError("Evaluation needs at least one transition")
    if not all(math.isfinite(value) for value in log_scores):
        raise ValueError("Non-finite log probability in evaluation")

    nll = -sum(log_scores) / len(log_scores)
    return {
        "nll": nll,
        "perplexity": math.exp(nll),
    }


def evaluate_heldout(
    model: NGramPrior,
    training_records: Sequence[NGramCorpusRecord],
    heldout_records: Sequence[NGramCorpusRecord],
    vocabularies: Vocabularies,
) -> dict:
    """Evaluate a trained model and baselines on identical held-out transitions.

    The uniform surrogate assumes independent, equally likely tokens for
    each selected stream, including that stream's unknown-token bucket.
    The real NullPrior remains an unnormalized constant-score prior.
    """
    train = tuple(training_records)
    heldout = tuple(heldout_records)
    if not train or not heldout:
        raise ValueError("Training and held-out records must be nonempty")

    train_ids = {record.piece_id for record in train}
    heldout_ids = {record.piece_id for record in heldout}
    if len(train_ids) != len(train) or len(heldout_ids) != len(heldout):
        raise ValueError("Duplicate piece IDs in split")
    if train_ids & heldout_ids:
        raise ValueError("Training and held-out pieces overlap")

    if model.vocabulary_tokens is not None:
        model.validate_vocabularies(vocabularies)

    unigram = NGramPrior.from_vocabularies(
        vocabularies,
        config=NGramConfig(
            order=1,
            alpha=model.config.alpha,
            streams=model.config.streams,
        ),
        provenance="held-out unigram baseline",
    )
    unigram.fit([record.states for record in train])

    queries = tuple(
        query
        for record in heldout
        for query in build_transition_examples(
            record.states,
            history_len=max(1, model.config.order - 1),
            future_hint_len=0,
        )
    )
    if not queries:
        raise ValueError("No held-out transitions")

    model_metrics = _metrics(model.logp_next_batch(queries))
    unigram_metrics = _metrics(unigram.logp_next_batch(queries))

    # NGramPrior uses one additional unknown-token bucket per stream.
    uniform_nll = sum(
        math.log(model.vocabulary_sizes[stream] + 1)
        for stream in model.config.streams
    )
    uniform_metrics = {
        "nll": uniform_nll,
        "perplexity": math.exp(uniform_nll),
    }

    # Raw NullPrior is intentionally not reported as normalized NLL/PPL.
    null_scores = NullPrior().logp_next_batch(queries)
    if any(score != 0.0 for score in null_scores):
        raise ValueError("Expected neutral NullPrior log score of zero")

    return {
        "report_version": 1,
        "evaluation": "held_out_next_state",
        "training_piece_ids": [r.piece_id for r in train],
        "heldout_piece_ids": [r.piece_id for r in heldout],
        "heldout_transitions": len(queries),
        "streams": list(model.config.streams),
        "model_order": model.config.order,
        "alpha": model.config.alpha,
        "vocabulary_sizes": dict(model.vocabulary_sizes),
        "provenance": model.provenance,
        "models": {
            "ngram": model_metrics,
            "unigram": unigram_metrics,
            "null_prior_uniform_surrogate": uniform_metrics,
            "null_prior_raw": {
                "log_score_per_transition": 0.0,
                "normalized": False,
                "nll": None,
                "perplexity": None,
            },
        },
        "notes": [
            "All probabilistic baselines score identical transitions and streams.",
            "Uniform surrogate uses vocabulary_size + 1 outcomes per stream, including UNK.",
            "NullPrior itself returns zero and is not a normalized probability model.",
            "Perplexity is exp(mean NLL) for the joint selected-stream target.",
            "Heuristic MIDI labels and the small song-level split limit generalization.",
        ],
    }


def save_report(report: dict, path: str | Path) -> Path:
    """Write a reproducible JSON evaluation report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return destination
