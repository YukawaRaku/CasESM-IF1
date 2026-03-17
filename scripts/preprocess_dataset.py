#!/usr/bin/env python
from cas_if1.config import load_yaml
from cas_if1.data.preprocess import PreprocessConfig, preprocess_dataset


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Cas structures into ESM-IF1 JSONL records.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config.")
    parser.add_argument("--input-dir", type=str, required=True, help="Raw input directory from fetch step.")
    parser.add_argument("--output-dir", type=str, required=True, help="Processed output directory.")
    args = parser.parse_args()

    cfg_dict = load_yaml(args.config) if args.config else {}
    config = PreprocessConfig.from_dict(cfg_dict)
    preprocess_dataset(input_dir=args.input_dir, output_dir=args.output_dir, config=config)


if __name__ == "__main__":
    main()

