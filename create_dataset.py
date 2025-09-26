#!/usr/bin/env python3
"""Collect images datasets into a single directory using sequential PNG filenames."""
from __future__ import annotations

import shutil
from pathlib import Path

PNG_SUFFIX = ".png"
DIGIT_WIDTH = 5


def find_next_index(dataset_root: Path) -> int:
    numbers = []
    for file in dataset_root.glob(f"*{PNG_SUFFIX}"):
        stem = file.stem
        if stem.isdigit():
            numbers.append(int(stem))
    return max(numbers, default=0) + 1


def collect_source_dirs(source_root: Path, dataset_root: Path) -> list[Path]:
    # Include legacy numbered directories in the dataset as additional sources so they get flattened.
    legacy_dirs = [path for path in dataset_root.iterdir() if path.is_dir()]
    image_dirs = [path for path in source_root.rglob("images") if path.is_dir()]
    return sorted(set(legacy_dirs + image_dirs), key=lambda p: p.as_posix())


def move_images(source_dir: Path, dataset_root: Path, starting_index: int) -> tuple[int, int]:
    next_index = starting_index
    moved = 0

    for image_path in sorted(source_dir.iterdir(), key=lambda p: p.as_posix()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() != PNG_SUFFIX:
            continue

        target = dataset_root / f"{next_index:0{DIGIT_WIDTH}d}{PNG_SUFFIX}"
        while target.exists():
            next_index += 1
            target = dataset_root / f"{next_index:0{DIGIT_WIDTH}d}{PNG_SUFFIX}"

        shutil.move(str(image_path), str(target))
        next_index += 1
        moved += 1

    # Remove the source directory if it is now empty and not the dataset root itself.
    if moved > 0:
        try:
            source_dir.rmdir()
        except OSError:
            pass

    return next_index, moved


def main() -> None:
    project_root = Path(__file__).resolve().parent
    source_root = project_root / "3Dbubbleflowrender"
    dataset_root = project_root / "dataset_bubble"

    if not source_root.exists():
        raise SystemExit(f"Source directory not found: {source_root}")

    dataset_root.mkdir(parents=True, exist_ok=True)

    sources = collect_source_dirs(source_root, dataset_root)
    if not sources:
        print("No new images directories found.")
        return

    next_index = find_next_index(dataset_root)
    total_moved = 0

    for source_dir in sources:
        if source_dir == dataset_root:
            continue

        next_index, moved = move_images(source_dir, dataset_root, next_index)
        if moved == 0:
            continue

        rel_dir = source_dir.relative_to(project_root)
        print(f"Moved {moved} PNG files from {rel_dir}")
        total_moved += moved

    if total_moved == 0:
        print("No new images directories found.")
    else:
        rel_dataset = dataset_root.relative_to(project_root)
        print(f"Moved {total_moved} PNG files into {rel_dataset}.")


if __name__ == "__main__":
    main()
