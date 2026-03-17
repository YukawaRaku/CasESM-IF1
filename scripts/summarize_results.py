#!/usr/bin/env python
from cas_if1.eval.runner import summarize_main


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize evaluation outputs into plots and CSVs.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory from evaluate.py.")
    parser.add_argument("--output-dir", type=str, required=True, help="Summary output directory.")
    args = parser.parse_args()

    summarize_main(results_dir=args.results_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

