#!/usr/bin/env python
from cas_if1.eval.runner import evaluate_main


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate baseline and finetuned ESM-IF1 models.")
    parser.add_argument("--test-jsonl", type=str, required=True, help="Held-out test JSONL.")
    parser.add_argument("--finetuned-checkpoint", type=str, required=True, help="Finetuned checkpoint.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    args = parser.parse_args()

    evaluate_main(
        test_jsonl=args.test_jsonl,
        finetuned_checkpoint=args.finetuned_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()

