
"""Train and evaluate grouped key-chord N-grams."""

import argparse
import math
from pathlib import Path

from aimusic.core.config import StyleConfig
from aimusic.core.vocab import build_default_vocabularies
from aimusic.ml.annotate import annotate_beat_grid
from aimusic.ml.dataset import build_transition_examples
from aimusic.ml.grouped_ngram import GroupedNGramPrior
from aimusic.ml.midi_ingest import parse_midi_beats


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("command", choices=["train"])
    parser.add_argument("--midi-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--held-out", type=int, default=2)

    args = parser.parse_args()

    root = Path(args.midi_dir)

    files = sorted(root.rglob("*.mid")) + sorted(
        root.rglob("*.midi")
    )

    if not 1 <= args.held_out < len(files):
        parser.error("Need more MIDI files than held-out files")

    vocab = build_default_vocabularies(StyleConfig())

    pieces = [
        tuple(
            annotate_beat_grid(
                parse_midi_beats(path),
                vocab,
            )
        )
        for path in files
    ]

    train_pieces = pieces[:-args.held_out]
    test_pieces = pieces[-args.held_out:]

    model = GroupedNGramPrior.from_vocabularies(
        vocab,
        order=args.order,
        alpha=args.alpha,
        provenance=str(root.resolve()),
    )

    model.fit(train_pieces)
    model.save(args.output)

    # Evaluate the serialized model, not just the in-memory one.
    loaded = GroupedNGramPrior.load(args.output)

    queries = [
        query
        for piece in test_pieces
        for query in build_transition_examples(
            piece,
            history_len=max(1, args.order - 1),
            future_hint_len=0,
        )
    ]

    if not queries:
        parser.error("No held-out transitions found")

    scores = loaded.logp_next_batch(queries)

    nll = -sum(scores) / len(scores)
    perplexity = math.exp(nll)

    print("\n=== Grouped NGramPrior Results ===")
    print(f"Saved model: {Path(args.output).resolve()}")
    print(f"Training pieces: {len(train_pieces)}")
    print(f"Test pieces: {len(test_pieces)}")
    print(f"Held-out transitions: {len(queries)}")
    print(f"N-gram order: {args.order}")
    print(f"Negative log-likelihood: {nll:.6f}")
    print(f"Perplexity: {perplexity:.6f}")


if __name__ == "__main__":
    main()
