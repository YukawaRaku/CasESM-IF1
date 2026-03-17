#!/usr/bin/env python
from cas_if1.config import load_yaml
from cas_if1.train.engine import train_main


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune ESM-IF1 for Cas inverse folding.")
    parser.add_argument("--config", type=str, required=True, help="Training YAML config.")
    parser.add_argument("--train-jsonl", type=str, required=True, help="Training records.")
    parser.add_argument("--val-jsonl", type=str, required=True, help="Validation records.")
    parser.add_argument("--output-dir", type=str, required=True, help="Run output directory.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    train_main(
        config=config,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()

