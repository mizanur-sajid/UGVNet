"""Scan an entire dataset and report issues without starting training."""

from __future__ import annotations

import argparse
from pathlib import Path

from ugvnet.data_audit import audit_dataset, enforce_audit_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an UGVNet dataset")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/dataset_audit.json"),
    )
    parser.add_argument(
        "--policy",
        choices=("strict", "warn"),
        default="strict",
        help="Strict stops on critical issues; warn reports them and exits normally.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Image-scanning workers; zero selects automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_dataset(
        args.data_dir,
        report_path=args.report,
        workers=args.workers or None,
    )
    enforce_audit_policy(report, args.policy)
    print("Dataset audit completed. Training was not started.")


if __name__ == "__main__":
    main()
