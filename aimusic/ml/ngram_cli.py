
"""Train and evaluate the individual NGramPrior."""

import argparse
import math
from pathlib import Path

from aimusic.core.config import StyleConfig
from aimusic.core.vocab import build_default_vocabularies
from aimusic.ml.dataset import build_transition_examples
from aimusic.ml.ngram_corpus import (
    load_ngram_corpus,
    corpus_provenance,
)
from aimusic.ml.ngram_prior import NGramConfig, NGramPrior


def main():
    parser = argparse.ArgumentParser(
        description="Train an individual musical NGramPrior"
    )

    parser.add_argument("command", choices=["train"])
    parser.add_argument("--midi-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--streams",
        nargs="+",
        default=["chord", "key"],
    )
    parser.add_argument("--held-out", type=int, default=2)

    args = parser.parse_args()

    root = Path(args.midi_dir)

    # Build the same vocabularies used for music generation.
    vocabularies = build_default_vocabularies(
        StyleConfig()
    )

    # Load MIDI files through the canonical corpus pipeline.
    # Each song remains an independent sequence of BeatStates.
    try:
        records = load_ngram_corpus(root, vocabularies)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))

    if not 1 <= args.held_out < len(records):
        parser.error(
            "Held-out size must be at least 1 and smaller "
            "than the number of MIDI songs"
        )

    # Preserve the ordering of each song's structural states.
    pieces = [record.states for record in records]

    # Split by song to prevent train/test leakage.
    training_records = records[:-args.held_out]
    evaluation_records = records[-args.held_out:]

    train_pieces = [record.states for record in training_records]
    test_pieces = [record.states for record in evaluation_records]

    # Record the exact training sources and evaluation split.
    metadata = corpus_provenance(training_records)
    metadata["held_out_piece_ids"] = [
        record.piece_id for record in evaluation_records
    ]

    model = NGramPrior.from_vocabularies(
        vocabularies,
        config=NGramConfig(
            order=args.order,
            alpha=args.alpha,
            streams=tuple(args.streams),
        ),
        provenance=metadata,
    )

    # Train and serialize the model.
    model.fit(train_pieces)
    model.save(args.output)

    # Evaluate the serialized model, not only the in-memory one.
    loaded_model = NGramPrior.load(args.output)

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

    scores = loaded_model.logp_next_batch(queries)

    nll = -sum(scores) / len(scores)
    perplexity = math.exp(nll)

    print("\n=== NGramPrior Results ===")
    print(f"Saved model: {Path(args.output).resolve()}")
    print(f"Training pieces: {len(train_pieces)}")
    print(f"Test pieces: {len(test_pieces)}")
    print(f"Held-out transitions: {len(queries)}")
    print(f"Streams: {args.streams}")
    print(f"N-gram order: {args.order}")
    print(f"Negative log-likelihood: {nll:.6f}")
    print(f"Perplexity: {perplexity:.6f}")


if __name__ == "__main__":
    main()
