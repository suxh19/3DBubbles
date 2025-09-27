#!/usr/bin/env python3
"""Collect images datasets into a single directory using sequential PNG filenames."""
from __future__ import annotations

import shutil
from pathlib import Path

PNG_SUFFIX = ".png"
DIGIT_WIDTH = 5
PROCESSED_LOG = ".processed_dirs.log"


def find_next_index(dataset_root: Path) -> int:
    numbers = []
    for file in dataset_root.glob(f"*{PNG_SUFFIX}"):
        stem = file.stem
        if stem.isdigit():
            numbers.append(int(stem))
    return max(numbers, default=0) + 1


def load_processed(dataset_root: Path) -> set[str]:
    log_path = dataset_root / PROCESSED_LOG
    if not log_path.exists():
        return set()
    with log_path.open("r", encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}


def append_processed(dataset_root: Path, processed_ids: list[str]) -> None:
    if not processed_ids:
        return
    log_path = dataset_root / PROCESSED_LOG
    with log_path.open("a", encoding="utf-8") as fh:
        for identifier in processed_ids:
            fh.write(f"{identifier}\n")


def collect_image_dirs(source_root: Path) -> list[Path]:
    return sorted(
        [path for path in source_root.rglob("images") if path.is_dir()],
        key=lambda p: p.as_posix(),
    )


def copy_images(source_dir: Path, dataset_root: Path, starting_index: int) -> tuple[int, int]:
    next_index = starting_index
    copied = 0

    for image_path in sorted(source_dir.iterdir(), key=lambda p: p.as_posix()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() != PNG_SUFFIX:
            continue

        target = dataset_root / f"{next_index:0{DIGIT_WIDTH}d}{PNG_SUFFIX}"
        while target.exists():
            next_index += 1
            target = dataset_root / f"{next_index:0{DIGIT_WIDTH}d}{PNG_SUFFIX}"

        shutil.copy2(image_path, target)
        next_index += 1
        copied += 1

    return next_index, copied


def main() -> None:
    project_root = Path(__file__).resolve().parent
    source_root = project_root / "3Dbubbleflowrender"
    dataset_root = project_root / "dataset_bubble"

    if not source_root.exists():
        raise SystemExit(f"Source directory not found: {source_root}")

    dataset_root.mkdir(parents=True, exist_ok=True)

    processed = load_processed(dataset_root)
    sources = collect_image_dirs(source_root)
    if not sources:
        print("No new images directories found.")
        return

    next_index = find_next_index(dataset_root)
    total_copied = 0
    newly_processed: list[str] = []

    for source_dir in sources:
        source_id = str(source_dir.resolve())
        if source_id in processed:
            continue

        next_index, copied = copy_images(source_dir, dataset_root, next_index)
        if copied == 0:
            continue

        rel_dir = source_dir.relative_to(project_root)
        print(f"Copied {copied} PNG files from {rel_dir}")
        total_copied += copied
        processed.add(source_id)
        newly_processed.append(source_id)

    append_processed(dataset_root, newly_processed)

    if total_copied == 0:
        print("No new images directories found.")
    else:
        rel_dataset = dataset_root.relative_to(project_root)
        print(f"Copied {total_copied} PNG files into {rel_dataset}.")


if __name__ == "__main__":
    main()
