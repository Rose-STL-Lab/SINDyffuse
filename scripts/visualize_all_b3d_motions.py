#!/usr/bin/env python3
"""Render Rajagopal mesh PDF montages for every Nimble B3D motion.

Uses the same settings as ``visualize_b3d_motion.py`` and writes both camera
presets per motion:

  results/viz/{motion_id}.pdf
  results/viz/{motion_id}_flat.pdf

Examples::

  python scripts/visualize_all_b3d_motions.py
  python scripts/visualize_all_b3d_motions.py --num_workers 16
  python scripts/visualize_all_b3d_motions.py --skip_existing
  python scripts/visualize_all_b3d_motions.py --max_motions 5
"""

from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

warnings.filterwarnings(
    "ignore",
    message=r"A NumPy version.*SciPy",
    category=UserWarning,
)

from common.paths import nimble_b3d_dir, resolve_data_root, resolve_repo_path, results_dir
from visualize_b3d_motion import (
    DATA_ROOT,
    NUM_POSES,
    render_b3d_motion_pdf,
)

VIEWS = ("default", "flat")
DEFAULT_NUM_WORKERS = 16


@dataclass(frozen=True)
class RenderJob:
    motion_id: str
    data_root: str
    output_dir: str
    num_poses: int
    skip_existing: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render default and flat Rajagopal mesh PDFs for all B3D motions.",
    )
    parser.add_argument(
        "--data_root",
        default=DATA_ROOT,
        help="Dataset root containing nimble_b3d/ (default: datasets/HumanML3D)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for PDF output (default: results/viz/)",
    )
    parser.add_argument(
        "--num_poses",
        type=int,
        default=NUM_POSES,
        help="Number of evenly spaced poses in each montage",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=(
            "Parallel worker processes (default: 16). "
            "Uses processes rather than threads for CPU-bound rendering."
        ),
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip motions whose PDFs already exist for all requested views",
    )
    parser.add_argument(
        "--max_motions",
        type=int,
        default=None,
        help="Process at most this many motions (useful for smoke tests)",
    )
    return parser.parse_args()


def _list_b3d_ids(b3d_dir: Path) -> list[str]:
    if not b3d_dir.is_dir():
        raise FileNotFoundError(f"Missing B3D cache directory: {b3d_dir}")
    ids = sorted(p.stem for p in b3d_dir.glob("*.b3d"))
    if not ids:
        raise FileNotFoundError(f"No .b3d files found under {b3d_dir}")
    return ids


def _output_path(output_dir: Path, motion_id: str, view: str) -> Path:
    if view == "default":
        return output_dir / f"{motion_id}.pdf"
    return output_dir / f"{motion_id}_{view}.pdf"


def _motion_complete(output_dir: Path, motion_id: str, views: tuple[str, ...]) -> bool:
    return all(_output_path(output_dir, motion_id, view).is_file() for view in views)


def _render_one_motion(job: RenderJob) -> dict[str, Any]:
    output_dir = Path(job.output_dir)
    saved: list[str] = []
    skipped_views: list[str] = []
    errors: list[str] = []

    for view in VIEWS:
        out_path = _output_path(output_dir, job.motion_id, view)
        if job.skip_existing and out_path.is_file():
            skipped_views.append(view)
            continue
        try:
            path = render_b3d_motion_pdf(
                motion_id=job.motion_id,
                data_root=job.data_root,
                output_pdf=out_path,
                num_poses=job.num_poses,
                view=view,
            )
            saved.append(str(path))
        except Exception as exc:
            errors.append(f"{view}: {exc}")

    if errors:
        status = "error"
    elif saved:
        status = "ok"
    else:
        status = "skipped"

    return {
        "motion_id": job.motion_id,
        "status": status,
        "saved": saved,
        "skipped_views": skipped_views,
        "errors": errors,
    }


def _run_sequential(
    jobs: list[RenderJob],
    *,
    total_motions: int,
    motions_skipped_upfront: int,
) -> tuple[int, int, list[tuple[str, str]]]:
    rendered = 0
    skipped = motions_skipped_upfront
    failed: list[tuple[str, str]] = []

    for idx, job in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] {job.motion_id}")
        result = _render_one_motion(job)
        if result["status"] == "skipped":
            skipped += 1
            print("  skip (existing)")
            continue
        if result["skipped_views"]:
            for view in result["skipped_views"]:
                print(f"  skip {view}")
        for path in result["saved"]:
            rendered += 1
            print(f"  saved {path}")
        for message in result["errors"]:
            failed.append((job.motion_id, message))
            print(f"  failed {message}")

    return rendered, skipped, failed


def _run_parallel(
    jobs: list[RenderJob],
    *,
    num_workers: int,
    total_motions: int,
    motions_skipped_upfront: int,
) -> tuple[int, int, list[tuple[str, str]]]:
    rendered = 0
    skipped = motions_skipped_upfront
    failed: list[tuple[str, str]] = []
    completed = 0

    pool_kwargs: dict[str, Any] = {"max_workers": max(1, int(num_workers))}
    if sys.version_info >= (3, 11):
        pool_kwargs["max_tasks_per_child"] = 4

    with ProcessPoolExecutor(**pool_kwargs) as ex:
        futures = {ex.submit(_render_one_motion, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            completed += 1
            try:
                result = fut.result()
            except Exception as exc:
                failed.append((job.motion_id, str(exc)))
                print(f"[{completed}/{len(jobs)}] failed {job.motion_id}: {exc}")
                continue

            if result["status"] == "skipped":
                skipped += 1
                print(f"[{completed}/{len(jobs)}] skip {job.motion_id} (existing)")
                continue

            rendered += len(result["saved"])
            if result["errors"]:
                for message in result["errors"]:
                    failed.append((job.motion_id, message))
                print(
                    f"[{completed}/{len(jobs)}] partial {job.motion_id}: "
                    f"{len(result['saved'])} saved, {len(result['errors'])} failed"
                )
            else:
                print(
                    f"[{completed}/{len(jobs)}] saved {job.motion_id} "
                    f"({len(result['saved'])} pdfs)"
                )

    return rendered, skipped, failed


def main() -> None:
    args = _parse_args()
    data_root = resolve_data_root(args.data_root)
    b3d_dir = nimble_b3d_dir(data_root)
    motion_ids = _list_b3d_ids(b3d_dir)

    if args.max_motions is not None:
        motion_ids = motion_ids[: max(0, int(args.max_motions))]

    if args.output_dir is None or str(args.output_dir).strip() == "":
        output_dir = results_dir() / "viz"
    else:
        output_dir = resolve_repo_path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    num_poses = int(args.num_poses)
    num_workers = max(1, int(args.num_workers))
    motions_skipped_upfront = 0
    jobs: list[RenderJob] = []

    for motion_id in motion_ids:
        if args.skip_existing and _motion_complete(output_dir, motion_id, VIEWS):
            motions_skipped_upfront += 1
            continue
        jobs.append(
            RenderJob(
                motion_id=motion_id,
                data_root=str(data_root),
                output_dir=str(output_dir),
                num_poses=num_poses,
                skip_existing=bool(args.skip_existing),
            )
        )

    total = len(motion_ids)
    if motions_skipped_upfront:
        print(f"Skipping {motions_skipped_upfront}/{total} motions with existing PDFs")

    if not jobs:
        print(
            f"Done. motions={total}, pdfs_rendered=0, "
            f"motions_skipped={motions_skipped_upfront}, failures=0"
        )
        return

    if num_workers == 1:
        rendered, skipped, failed = _run_sequential(
            jobs,
            total_motions=total,
            motions_skipped_upfront=motions_skipped_upfront,
        )
    else:
        print(f"Rendering {len(jobs)} motions with {num_workers} worker processes")
        rendered, skipped, failed = _run_parallel(
            jobs,
            num_workers=num_workers,
            total_motions=total,
            motions_skipped_upfront=motions_skipped_upfront,
        )

    print(
        f"Done. motions={total}, pdfs_rendered={rendered}, motions_skipped={skipped}, "
        f"failures={len(failed)}"
    )
    if failed:
        for motion_id, message in failed[:10]:
            print(f"  {motion_id}: {message}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
