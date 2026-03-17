#!/usr/bin/env python
from cas_if1.models.esm_if1 import inference_main


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inverse fold structures with a finetuned ESM-IF1 model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint produced by training.")
    parser.add_argument("--input", type=str, required=True, help="Input structure file or directory.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--num-samples", type=int, default=8, help="Samples per structure.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    args = parser.parse_args()

    inference_main(
        checkpoint_path=args.checkpoint,
        input_path=args.input,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        temperature=args.temperature,
        device=args.device,
    )


if __name__ == "__main__":
    main()

