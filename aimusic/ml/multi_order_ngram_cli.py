
"""Compare grouped N-grams against multi-order interpolation."""

import argparse
import math
from pathlib import Path

from aimusic.core.config import StyleConfig
from aimusic.core.vocab import build_default_vocabularies
from aimusic.ml.annotate import annotate_beat_grid
from aimusic.ml.dataset import build_transition_examples
from aimusic.ml.midi_ingest import parse_midi_beats
from aimusic.ml.grouped_ngram import GroupedNGramPrior
from aimusic.ml.multi_order_ngram import MultiOrderNGramPrior


def evaluate(model, queries):
    scores = model.logp_next_batch(queries)
    nll = -sum(scores) / len(scores)
    return nll, math.exp(nll)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi-dir", required=True)
    parser.add_argument("--unigram", required=True)
    parser.add_argument("--bigram", required=True)
    parser.add_argument("--trigram", required=True)
    parser.add_argument("--held-out", type=int, default=2)
    parser.add_argument("--strength", type=float, default=20.0)
    args = parser.parse_args()

    unigram = GroupedNGramPrior.load(args.unigram)
    bigram = GroupedNGramPrior.load(args.bigram)
    trigram = GroupedNGramPrior.load(args.trigram)

    multi = MultiOrderNGramPrior(
        unigram, bigram, trigram, args.strength
    )

    root = Path(args.midi_dir)
    files = sorted(
        [*root.rglob("*.mid"), *root.rglob("*.midi")]
    )

    if not 1 <= args.held_out < len(files):
        parser.error("Invalid held-out size")

    vocab = build_default_vocabularies(StyleConfig())

    pieces = [
        tuple(annotate_beat_grid(parse_midi_beats(path), vocab))
        for path in files[-args.held_out:]
    ]

    queries = [
        query
        for piece in pieces
        for query in build_transition_examples(
            piece,
            history_len=2,
            future_hint_len=0,
        )
    ]

    if not queries:
        parser.error("No held-out transitions")

    print("\n=== Multi-Order Adaptive Comparison ===")
    print(f"Transitions: {len(queries)}")
    print(f"Strength: {args.strength}")
    print(f"{'Model':<20} {'NLL':>12} {'Perplexity':>15}")

    for name, model in (
        ("Grouped unigram", unigram),
        ("Grouped bigram", bigram),
        ("Grouped trigram", trigram),
        ("Multi-order adaptive", multi),
    ):
        nll, ppl = evaluate(model, queries)
        print(f"{name:<20} {nll:>12.6f} {ppl:>15.6f}")


if __name__ == "__main__":
    main()
