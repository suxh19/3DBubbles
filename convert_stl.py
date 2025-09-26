"""Batch STL-to-image converter.

This utility wraps ``tovoxel.convert_file`` so we can render STL files into PNGs.
It supports converting a single file or walking directories. For each flow folder
(e.g. ``3Dbubbleflowrender/20250926T124110-480749-flow0000``) an ``images``
subfolder is created where the rendered PNGs are stored. If that folder already
exists we assume the flow has been processed and skip it.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List

try:  # Prefer the name requested by the user.
    import tovoxel  # type: ignore
except ImportError:  # Fall back to the original module name if available.
    import stltovoxel as tovoxel  # type: ignore

DEFAULT_ROOT = Path("3Dbubbleflowrender")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert STL files to PNGs using tovoxel.convert_file",
    )
    default_jobs = 8
    parser.add_argument(
        "targets",
        nargs="*",
        default=[str(DEFAULT_ROOT)],
        help="Files or directories to convert (defaults to 3Dbubbleflowrender)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1000000,
        help="Voxel resolution passed to tovoxel.convert_file (default: 1000000)",
    )
    parser.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="Disable parallel mode in tovoxel.convert_file",
    )
    parser.set_defaults(parallel=True)
    parser.add_argument(
        "--voxel-size",
        nargs=3,
        type=float,
        default=[0.005, 0.005, 0.1],
        metavar=("X", "Y", "Z"),
        help="Voxel size in each dimension (default: 0.005 0.005 0.1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Convert even if an images folder already exists",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=default_jobs,
        metavar="N",
        help=f"Number of worker threads used to convert STL files (default: {default_jobs})",
    )
    return parser.parse_args()


def collect_targets(targets: Iterable[str]) -> Dict[Path, List[Path]]:
    """Group STL files by their parent directory."""
    grouped: Dict[Path, List[Path]] = {}
    for target in targets:
        path = Path(target).resolve()
        if path.is_file() and path.suffix.lower() == ".stl":
            grouped.setdefault(path.parent, []).append(path)
            continue
        if path.is_dir():
            for stl_file in sorted(path.rglob("*.stl")):
                grouped.setdefault(stl_file.parent, []).append(stl_file)
            continue
        print(f"[warn] Skipping unknown target: {path}")
    return grouped


def convert_directory(
    flow_dir: Path,
    stl_files: Iterable[Path],
    resolution: int,
    parallel: bool,
    voxel_size: List[float],
    force: bool,
    executor: ThreadPoolExecutor | None,
) -> List[Future[None]]:
    images_dir = flow_dir / "images"
    if images_dir.exists() and not force:
        print(f"[skip] {flow_dir} already has an images folder")
        return []

    images_dir.mkdir(exist_ok=True)
    unique_paths = sorted({Path(stl_path).resolve() for stl_path in stl_files})
    futures: List[Future[None]] = []
    for stl_path in unique_paths:
        output_path = images_dir / f"{stl_path.stem}.png"
        if executor is None:
            convert_single(
                stl_path,
                output_path,
                resolution,
                parallel,
                voxel_size,
            )
        else:
            futures.append(
                executor.submit(
                    convert_single,
                    stl_path,
                    output_path,
                    resolution,
                    parallel,
                    voxel_size,
                )
            )
    return futures


def convert_single(
    stl_path: Path,
    output_path: Path,
    resolution: int,
    parallel: bool,
    voxel_size: List[float],
) -> None:
    print(f"[convert] {stl_path} -> {output_path}")
    tovoxel.convert_file(
        str(stl_path),
        str(output_path),
        resolution=resolution,
        parallel=parallel,
        voxel_size=voxel_size,
    )


def main() -> None:
    args = parse_args()
    groups = collect_targets(args.targets)
    if not groups:
        print("[info] Nothing to convert")
        return

    if args.jobs < 1:
        print(f"[warn] Invalid --jobs value {args.jobs}; falling back to a single worker")
        max_workers = 1
    else:
        max_workers = args.jobs

    futures: List[Future[None]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for flow_dir, stl_files in sorted(groups.items()):
            futures.extend(
                convert_directory(
                    flow_dir,
                    stl_files,
                    args.resolution,
                    args.parallel,
                    args.voxel_size,
                    args.force,
                    executor,
                )
            )

        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
