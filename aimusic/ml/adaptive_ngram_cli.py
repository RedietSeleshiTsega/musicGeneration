
"""Evaluate individual, grouped, and adaptive models together."""

import argparse
import math
from pathlib import Path

from aimusic.core.config import StyleConfig
from aimusic.core.vocab import build_default_vocabularies
from aimusic.ml.annotate import annotate_beat_grid
from aimusic.ml.dataset import build_transition_examples
from aimusic.ml.midi_ingest import parse_midi_beats
from aimusic.ml.ngram_prior import NGramPrior
from aimusic.ml.grouped_ngram import GroupedNGramPrior
from aimusic.ml.adaptive_ngram import AdaptiveNGramPrior


def evaluate(model, queries):
    scores = model.logp_next_batch(queries)

    nll = -sum(scores) / len(scores)
    return nll, math.exp(nll)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--midi-dir", required=True)
    parser.add_argument("--individual", required=True)
    parser.add_argument("--grouped", required=True)
    parser.add_argument("--held-out", type=int, default=2)
    parser.add_argument("--strength", type=float, default=20.0)

    args = parser.parse_args()

    individual = NGramPrior.load(args.individual)
    grouped = GroupedNGramPrior.load(args.grouped)

    adaptive = AdaptiveNGramPrior(
        individual,
        grouped,
        strength=args.strength,
    )

    root = Path(args.midi_dir)

    files = sorted(root.rglob("*.mid")) + sorted(
        root.rglob("*.midi")
    )

    if not 1 <= args.held_out < len(files):
        parser.error("Invalid held-out size")

    vocab = build_default_vocabularies(StyleConfig())

    # Match the hold-out split used in both training commands.
    test_files = files[-args.held_out:]

    test_pieces = [
        tuple(
            annotate_beat_grid(
                parse_midi_beats(path),
                vocab,
            )
        )
        for path in test_files
    ]

    queries = [
        query
        for piece in test_pieces
        for query in build_transition_examples(
            piece,
            history_len=max(
                individual.config.order,
                grouped.order,
            ) - 1,
            future_hint_len=0,
        )
    ]

    if not queries:
        parser.error("No evaluation transitions found")

    print("\n=== Held-Out Model Comparison ===")
    print(f"Test files: {[p.name for p in test_files]}")
    print(f"Transitions: {len(queries)}")
    print(f"Adaptive strength: {args.strength}")
    print()
    print(f"{'Model':<15} {'NLL':>12} {'Perplexity':>15}")

    for name, model in (
        ("Individual", individual),
        ("Grouped", grouped),
        ("Adaptive", adaptive),
    ):
        nll, perplexity = evaluate(model, queries)

        print(
            f"{name:<15} {nll:>12.6f} "
            f"{perplexity:>15.6f}"
        )


if __name__ == "__main__":
    main()
