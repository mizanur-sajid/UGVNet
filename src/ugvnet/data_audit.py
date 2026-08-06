"""Pre-training dataset audit for ImageFolder-style classification datasets."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

REQUIRED_SPLITS = ("train", "validation", "test")
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
}


class DatasetAuditError(RuntimeError):
    """Raised when a dataset fails the selected audit policy."""


@dataclass(frozen=True)
class ImageRecord:
    path: str
    split: str
    class_name: str
    size_bytes: int
    width: int | None
    height: int | None
    image_format: str | None
    mode: str | None
    sha256: str | None
    error: str | None


def _inspect_image(item: tuple[Path, Path, str, str]) -> ImageRecord:
    path, root, split, class_name = item
    relative_path = path.relative_to(root).as_posix()
    size_bytes = 0
    try:
        payload = path.read_bytes()
        size_bytes = len(payload)
        digest = hashlib.sha256(payload).hexdigest()
        with Image.open(BytesIO(payload)) as image:
            width, height = image.size
            image_format = image.format or path.suffix.removeprefix(".").upper()
            mode = image.mode
            image.load()
        if width < 1 or height < 1:
            raise ValueError(f"invalid dimensions: {width}x{height}")
        return ImageRecord(
            path=relative_path,
            split=split,
            class_name=class_name,
            size_bytes=size_bytes,
            width=width,
            height=height,
            image_format=image_format,
            mode=mode,
            sha256=digest,
            error=None,
        )
    except Exception as error:  # noqa: BLE001 - decoders raise varied exceptions.
        return ImageRecord(
            path=relative_path,
            split=split,
            class_name=class_name,
            size_bytes=size_bytes,
            width=None,
            height=None,
            image_format=None,
            mode=None,
            sha256=None,
            error=f"{type(error).__name__}: {error}",
        )


def _discover_files(
    root: Path,
) -> tuple[list[tuple[Path, Path, str, str]], list[str], dict[str, list[str]]]:
    work: list[tuple[Path, Path, str, str]] = []
    ignored: list[str] = []
    classes: dict[str, list[str]] = {}
    missing_splits = [split for split in REQUIRED_SPLITS if not (root / split).is_dir()]
    if missing_splits:
        raise DatasetAuditError(
            "Missing required dataset folders: " + ", ".join(missing_splits)
        )

    for split in REQUIRED_SPLITS:
        split_path = root / split
        class_directories = sorted(
            path for path in split_path.iterdir() if path.is_dir()
        )
        classes[split] = [path.name for path in class_directories]
        for class_directory in class_directories:
            for path in sorted(class_directory.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    work.append((path, root, split, class_directory.name))
                else:
                    ignored.append(path.relative_to(root).as_posix())
    return work, ignored, classes


def _duplicate_groups(records: Iterable[ImageRecord]) -> list[list[ImageRecord]]:
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.sha256 is not None:
            by_hash[record.sha256].append(record)
    return [group for group in by_hash.values() if len(group) > 1]


def _group_paths(groups: Iterable[list[ImageRecord]]) -> list[list[str]]:
    return [[record.path for record in group] for group in groups]


def audit_dataset(
    data_dir: str | Path,
    *,
    report_path: str | Path | None = None,
    workers: int | None = None,
    progress_every: int = 1000,
) -> dict:
    """Decode and inspect every image in a three-split dataset.

    The returned report contains only aggregate data and issue paths, keeping
    its size practical even when the complete dataset contains many images.
    """

    started = time.perf_counter()
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise DatasetAuditError(f"Dataset root does not exist: {root}")
    work, ignored_files, split_classes = _discover_files(root)
    if not work:
        raise DatasetAuditError(f"No supported images found under: {root}")

    if workers is not None and workers < 0:
        raise ValueError("workers must be non-negative or None.")
    resolved_workers = workers or min(8, max(1, (os.cpu_count() or 2)))
    records: list[ImageRecord] = []
    print(
        f"\nDataset audit: decoding and hashing {len(work):,} images "
        f"with {resolved_workers} workers..."
    )
    with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
        for index, record in enumerate(executor.map(_inspect_image, work), start=1):
            records.append(record)
            if progress_every > 0 and (
                index % progress_every == 0 or index == len(work)
            ):
                print(f"  scanned {index:,}/{len(work):,} images")

    valid_records = [record for record in records if record.error is None]
    corrupt_records = [record for record in records if record.error is not None]
    duplicates = _duplicate_groups(valid_records)
    cross_split_duplicates = [
        group for group in duplicates if len({record.split for record in group}) > 1
    ]
    within_split_duplicates = [
        group for group in duplicates if len({record.split for record in group}) == 1
    ]

    class_counts: dict[str, dict[str, int]] = {}
    split_counts: dict[str, int] = {}
    empty_classes: dict[str, list[str]] = {}
    for split in REQUIRED_SPLITS:
        counts = Counter(
            record.class_name for record in records if record.split == split
        )
        class_counts[split] = {
            class_name: counts.get(class_name, 0) for class_name in split_classes[split]
        }
        split_counts[split] = sum(class_counts[split].values())
        empty_classes[split] = [
            class_name
            for class_name, count in class_counts[split].items()
            if count == 0
        ]

    class_names_consistent = (
        split_classes["train"] == split_classes["validation"] == split_classes["test"]
    )
    train_nonzero = [count for count in class_counts["train"].values() if count > 0]
    imbalance_ratio = (
        max(train_nonzero) / min(train_nonzero) if train_nonzero else float("inf")
    )
    widths = [record.width for record in valid_records if record.width is not None]
    heights = [record.height for record in valid_records if record.height is not None]
    resolutions = Counter(f"{record.width}x{record.height}" for record in valid_records)
    formats = Counter(record.image_format or "UNKNOWN" for record in valid_records)
    modes = Counter(record.mode or "UNKNOWN" for record in valid_records)
    total_bytes = sum(record.size_bytes for record in records)

    critical_issue_count = (
        len(corrupt_records)
        + len(cross_split_duplicates)
        + (0 if class_names_consistent else 1)
        + sum(len(names) for names in empty_classes.values())
    )
    recommendations: list[str] = []
    if corrupt_records:
        recommendations.append("Remove or replace every corrupt image.")
    if cross_split_duplicates:
        recommendations.append(
            "Remove exact duplicates that cross train/validation/test boundaries."
        )
    if within_split_duplicates:
        recommendations.append(
            "Review within-split duplicates to prevent repeated samples from biasing training."
        )
    if not class_names_consistent:
        recommendations.append(
            "Make class-folder names identical in train, validation, and test."
        )
    if imbalance_ratio >= 1.5:
        recommendations.append(
            "Keep balanced sampling enabled and report macro-F1 per class."
        )
    if ignored_files:
        recommendations.append(
            "Review ignored non-image files if they were expected to be training samples."
        )
    if not recommendations:
        recommendations.append("No critical dataset issues were detected.")

    report = {
        "schema_version": 1,
        "dataset_root": str(root),
        "scanned_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "summary": {
            "candidate_images": len(records),
            "valid_images": len(valid_records),
            "corrupt_images": len(corrupt_records),
            "ignored_files": len(ignored_files),
            "total_size_gib": round(total_bytes / 2**30, 3),
            "exact_duplicate_groups": len(duplicates),
            "within_split_duplicate_groups": len(within_split_duplicates),
            "cross_split_duplicate_groups": len(cross_split_duplicates),
            "critical_issue_count": critical_issue_count,
        },
        "split_image_counts": split_counts,
        "class_counts": class_counts,
        "split_classes": split_classes,
        "class_names_consistent": class_names_consistent,
        "empty_classes": empty_classes,
        "training_imbalance_ratio": round(imbalance_ratio, 4),
        "image_properties": {
            "width": {
                "min": min(widths) if widths else None,
                "median": statistics.median(widths) if widths else None,
                "max": max(widths) if widths else None,
            },
            "height": {
                "min": min(heights) if heights else None,
                "median": statistics.median(heights) if heights else None,
                "max": max(heights) if heights else None,
            },
            "formats": dict(formats.most_common()),
            "color_modes": dict(modes.most_common()),
            "most_common_resolutions": dict(resolutions.most_common(12)),
        },
        "issues": {
            "corrupt_images": [asdict(record) for record in corrupt_records],
            "cross_split_duplicates": _group_paths(cross_split_duplicates),
            "within_split_duplicates": _group_paths(within_split_duplicates),
            "ignored_file_examples": ignored_files[:100],
        },
        "recommendations": recommendations,
    }

    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(destination.resolve())
    print_audit_report(report)
    return report


def print_audit_report(report: dict, *, issue_examples: int = 8) -> None:
    """Print a compact, user-facing summary before training begins."""

    summary = report["summary"]
    properties = report["image_properties"]
    
    width = 72
    lines = []
    
    def add_sep(char_left, char_mid, char_right, char_line='─'):
        lines.append(char_left + char_line * width + char_right)
        
    def add_line(text):
        lines.append(f"│ {text:<{width-1}}│")

    lines.append("")
    add_sep('┌', '─', '┐')
    lines.append("│" + "UGVNet PRE-TRAINING DATASET AUDIT".center(width) + "│")
    add_sep('├', '─', '┤')

    def add_4col(k, v1, v2, v3):
        lines.append(f"│ {k:<16} │ {v1:>14} │ {v2:>16} │ {v3:>19} │")

    add_4col(
        "Images",
        f"{summary['candidate_images']:,} total",
        f"{summary['valid_images']:,} valid",
        f"{summary['corrupt_images']:,} corrupt"
    )
    add_4col("Ignored files", f"{summary['ignored_files']:,}", "", "")
    
    fmt_strs = [f"{k} ({v:,})" for k, v in properties["formats"].items()]
    mode_strs = [f"{k} ({v:,})" for k, v in properties["color_modes"].items()]
    add_4col("Formats", "", "", ", ".join(fmt_strs[:2]))
    add_4col("Color modes", "", "", ", ".join(mode_strs[:2]))

    add_line(f"{'Width (px)':<17}│ min {properties['width']['min']}  ·  median {properties['width']['median']}  ·  max {properties['width']['max']}")
    add_line(f"{'Height (px)':<17}│ min {properties['height']['min']}  ·  median {properties['height']['median']}  ·  max {properties['height']['max']}")
    add_line(f"{'Imbalance ratio':<17}│{f'{report["training_imbalance_ratio"]:.2f}x':>53}")
    add_line(f"{'Duplicate groups':<17}│ {summary['exact_duplicate_groups']:,} total      ·       {summary['cross_split_duplicate_groups']:,} crossing splits")
    add_sep('├', '─', '┤')

    all_classes = sorted({c for counts in report["class_counts"].values() for c in counts})
    header = f"{'Class splits':<26}" + "".join(f"{s:>15s}" for s in REQUIRED_SPLITS)
    add_line(header)
    for cls in all_classes:
        row = f"{cls[:26]:<26}" + "".join(f"{report['class_counts'][s].get(cls, 0):15,d}" for s in REQUIRED_SPLITS)
        add_line(row)
    add_line(f"{'TOTAL':<26}" + "".join(f"{report['split_image_counts'][s]:15,d}" for s in REQUIRED_SPLITS))
    add_sep('├', '─', '┤')

    issues = False
    if report["issues"]["corrupt_images"]:
        add_line("Corrupt-image examples:")
        for item in report["issues"]["corrupt_images"][:issue_examples]:
            add_line(f"  - {str(item['path'])[-65:]:>65}")
        issues = True

    if report["issues"]["cross_split_duplicates"]:
        if issues: add_line("")
        add_line("Cross-split duplicate examples:")
        for group in report["issues"]["cross_split_duplicates"][:issue_examples]:
            add_line("  - " + " | ".join(str(p)[-30:] for p in group)[:67])
        issues = True

    if report["recommendations"]:
        if issues: add_line("")
        add_line("Recommendations:")
        for rec in report["recommendations"]:
            add_line(f"  - {rec}")
        issues = True
    
    if issues:
        add_sep('├', '─', '┤')

    if "report_path" in report:
        add_line(f"Report: {report['report_path']}")
    
    status_icon = "✓" if summary["critical_issue_count"] == 0 else "⚠"
    status_text = "PASS" if summary["critical_issue_count"] == 0 else "ATTENTION REQUIRED"
    add_line(f"Status: {status_icon} {status_text}")
    
    add_sep('└', '─', '┘')
    lines.append("")

    for line in lines:
        print(line)


def enforce_audit_policy(report: dict, policy: str = "strict") -> None:
    """Stop before training when strict policy finds critical data problems."""

    normalized = policy.lower()
    if normalized not in {"strict", "warn", "off"}:
        raise ValueError("Audit policy must be 'strict', 'warn', or 'off'.")
    if normalized == "off":
        return
    critical = report["summary"]["critical_issue_count"]
    if critical > 0 and normalized == "strict":
        raise DatasetAuditError(
            f"Dataset audit found {critical} critical issue(s). "
            "Review dataset_audit.json, correct the dataset, and run again. "
            "Use policy='warn' only when you explicitly accept these risks."
        )
    if critical > 0:
        print(
            f"WARNING: continuing despite {critical} critical dataset issue(s) "
            "because audit policy is 'warn'."
        )
