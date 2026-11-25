"""Train and evaluate the individual NGramPrior."""

import argparse
from pathlib import Path

from aimusic.core.config import StyleConfig
from aimusic.core.vocab import build_default_vocabularies
from aimusic.ml.ngram_corpus import load_ngram_corpus, corpus_provenance
from aimusic.ml.ngram_evaluation import evaluate_heldout, save_report
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
    parser.add_argument(
        "--report",
        default=None,
        help="JSON report path; default: <output_stem>_heldout_report.json",
    )

    args = parser.parse_args()
    root = Path(args.midi_dir)
    vocabularies = build_default_vocabularies(StyleConfig())

    try:
        records = load_ngram_corpus(root, vocabularies)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))

    if not 1 <= args.held_out < len(records):
        parser.error("Held-out size must be at least 1 and smaller than the corpus")

    training_records = records[:-args.held_out]
    evaluation_records = records[-args.held_out:]
    train_pieces = [record.states for record in training_records]
    test_pieces = [record.states for record in evaluation_records]

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
    model.fit(train_pieces)
    model.save(args.output)

    # Evaluate the serialized model, not just the in-memory instance.
    loaded_model = NGramPrior.load(args.output)
    report = evaluate_heldout(
        loaded_model, training_records, evaluation_records, vocabularies
    )

    output = Path(args.output)
    report_path = Path(args.report) if args.report else output.with_name(
        output.stem + "_heldout_report.json"
    )
    save_report(report, report_path)

    results = report["models"]
    print("\n=== NGramPrior Held-Out Comparison ===")
    print(f"Saved model: {output.resolve()}")
    print(f"Saved JSON report: {report_path.resolve()}")
    print(f"Training pieces: {len(train_pieces)}")
    print(f"Test pieces: {len(test_pieces)}")
    print(f"Held-out transitions: {report['heldout_transitions']}")
    print(f"Streams: {args.streams}")
    print(f"N-gram order: {args.order}")
    for label in ("ngram", "unigram", "null_prior_uniform_surrogate"):
        metrics = results[label]
        print(
            f"{label}: NLL={metrics['nll']:.6f}, "
            f"perplexity={metrics['perplexity']:.6f}"
        )
    print("Actual NullPrior: neutral score 0 (unnormalized; no NLL/PPL)")


if __name__ == "__main__":
    main()
