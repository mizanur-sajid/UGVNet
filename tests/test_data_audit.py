from pathlib import Path
from shutil import copyfile

import pytest
from PIL import Image

from ugvnet.data_audit import DatasetAuditError, audit_dataset, enforce_audit_policy


def save_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), color=color).save(path)


def build_dataset(root: Path) -> None:
    for split in ("train", "validation", "test"):
        for class_name in ("benign", "malignant"):
            save_image(
                root / split / class_name / f"{split}_{class_name}.png",
                (20, 40, 60) if class_name == "benign" else (180, 30, 20),
            )


def test_audit_detects_corruption_and_cross_split_duplicates(
    tmp_path: Path,
) -> None:
    build_dataset(tmp_path)
    copyfile(
        tmp_path / "train" / "benign" / "train_benign.png",
        tmp_path / "test" / "benign" / "leaked.png",
    )
    corrupt = tmp_path / "validation" / "malignant" / "corrupt.jpg"
    corrupt.write_bytes(b"not a valid image")

    report_path = tmp_path / "audit.json"
    report = audit_dataset(
        tmp_path,
        report_path=report_path,
        workers=1,
        progress_every=0,
    )

    assert report_path.is_file()
    assert report["summary"]["corrupt_images"] == 1
    assert report["summary"]["cross_split_duplicate_groups"] >= 1
    assert report["summary"]["critical_issue_count"] >= 2
    with pytest.raises(DatasetAuditError):
        enforce_audit_policy(report, "strict")


def test_invalid_worker_count_is_rejected(tmp_path: Path) -> None:
    build_dataset(tmp_path)
    with pytest.raises(ValueError, match="workers must be non-negative"):
        audit_dataset(tmp_path, workers=-1, progress_every=0)


def test_clean_dataset_passes_strict_policy(tmp_path: Path) -> None:
    build_dataset(tmp_path)
    # Make every image byte-distinct to avoid intentional color duplicates.
    for index, path in enumerate(sorted(tmp_path.rglob("*.png"))):
        Image.new("RGB", (12 + index, 10), color=(index, 20, 30)).save(path)

    report = audit_dataset(tmp_path, workers=1, progress_every=0)

    assert report["summary"]["critical_issue_count"] == 0
    enforce_audit_policy(report, "strict")
