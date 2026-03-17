#!/usr/bin/env python
from cas_if1.config import load_yaml
from cas_if1.data.acquisition import FetchConfig, fetch_cas_dataset


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Cas protein structures and metadata from RCSB.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--max-results", type=int, default=None, help="Override maximum number of entries.")
    parser.add_argument("--download-format", choices=["pdb", "cif"], default=None, help="Structure format.")
    args = parser.parse_args()

    cfg_dict = load_yaml(args.config) if args.config else {}
    if args.max_results is not None:
        cfg_dict["max_results"] = args.max_results
    if args.download_format is not None:
        cfg_dict["download_format"] = args.download_format

    config = FetchConfig.from_dict(cfg_dict)
    fetch_cas_dataset(config=config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

