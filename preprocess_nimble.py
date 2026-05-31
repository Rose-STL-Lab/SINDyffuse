#!/usr/bin/env python3
"""Preprocess HumanML3D motion into Nimble B3D cache (one .b3d per motion).

Run from the SINDyffuse repo root, e.g.:

  python preprocess_nimble.py --num_workers 8
  python preprocess_nimble.py --max_motions 1 --num_workers 1   # smoke test

Defaults: ``datasets/HumanML3D`` for input and output; writes ``nimble_b3d/*.b3d``
and q-space Mean.npy / Std.npy under that subfolder. The bundled Rajagopal 2015
skeleton is the only model -- see ``nimble/skeleton_registry.py`` for the IK
mapping.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ``nimble.export`` is imported lazily inside workers to avoid eagerly loading
# the nimble skeleton in the parent process. ``compute_nimble_normalization_stats``
# only touches written B3D files (post-processing) so it's safe to import here.
from common.paths import NIMBLE_B3D_SUBDIR, default_humanml3d_root, nimble_b3d_dir
from datasets.hml3d_joints import default_joints_root, load_hml3d_joint_positions
from datasets.nimble_dataset import compute_nimble_normalization_stats
from datasets.splits import all_motion_ids


WorkItem = tuple[str, str, str, float, float, float, bool, str, str]


def _load_joints(
    hml_root: Path,
    sid: str,
    *,
    joint_source: str,
    joints_root: Path | None,
) -> np.ndarray | None:
    try:
        joints, _ = load_hml3d_joint_positions(
            hml_root,
            sid,
            joint_source=joint_source,  # type: ignore[arg-type]
            joints_root=joints_root,
        )
        return joints
    except FileNotFoundError:
        return None


def _link_or_copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    if sys.platform != "win32":
        try:
            os.symlink(src, dst, target_is_directory=True)
            return
        except OSError:
            pass
    shutil.copytree(src, dst)


def _process_one(args: WorkItem) -> dict:
    (
        sid,
        hml_root_s,
        out_root_s,
        fps,
        mass_kg,
        height_m,
        skip_existing,
        joint_source,
        joints_root_s,
    ) = args
    from nimble.export import export_motion_to_b3d

    hml_root = Path(hml_root_s)
    out_root = Path(out_root_s)
    joints_root = Path(joints_root_s) if joints_root_s else None
    out_b3d = nimble_b3d_dir(out_root) / f"{sid}.b3d"
    if skip_existing and out_b3d.is_file():
        return {"id": sid, "status": "skipped", "path": str(out_b3d)}

    joints = _load_joints(
        hml_root,
        sid,
        joint_source=joint_source,
        joints_root=joints_root,
    )
    if joints is None:
        return {"id": sid, "status": "error", "error": "missing or invalid motion"}

    try:
        stats, num_dofs = export_motion_to_b3d(
            joints,
            out_b3d,
            trial_name=sid,
            fps=fps,
            mass_kg=mass_kg,
            height_m=height_m,
        )
    except Exception as exc:
        return {"id": sid, "status": "error", "error": str(exc)}

    return {
        "id": sid,
        "status": "ok",
        "path": str(out_b3d),
        "num_dofs": int(num_dofs),
        "ik_stats": stats,
    }


def _motion_loss(row: dict) -> float | None:
    if row.get("status") != "ok":
        return None
    ik = row.get("ik_stats") or {}
    loss = ik.get("mean_fit_joints_loss", ik.get("mean_ik_error"))
    if loss is None:
        return None
    val = float(loss)
    return val if np.isfinite(val) else None


def _symlink_metadata(hml_root: Path, out_root: Path) -> None:
    if hml_root.resolve() == out_root.resolve():
        return
    src_texts = hml_root / "texts"
    if src_texts.is_dir():
        _link_or_copy_tree(src_texts, out_root / "texts")
    for name in ("train.txt", "val.txt", "test.txt"):
        src = hml_root / name
        dst = out_root / name
        if src.is_file() and not dst.exists():
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="HumanML3D → Nimble B3D preprocessing")
    default_root = default_humanml3d_root()
    parser.add_argument(
        "--hml_root",
        default=default_root,
        help="HumanML3D root (new_joints or new_joint_vecs; default datasets/HumanML3D)",
    )
    parser.add_argument(
        "--out_root",
        default=default_root,
        help=f"Dataset root; B3D motions go to <out_root>/{NIMBLE_B3D_SUBDIR}/",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--mass_kg", type=float, default=70.0)
    parser.add_argument("--height_m", type=float, default=1.75)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip motions that already have a .b3d in nimble_b3d/",
    )
    parser.add_argument(
        "--max_motions",
        type=int,
        default=0,
        help="Process at most this many motions (0 = all; useful for smoke tests)",
    )
    parser.add_argument(
        "--joint_source",
        choices=("auto", "joints", "new_joints"),
        default="auto",
        help="Joint XYZ input: auto prefers ./joints/ then new_joints/",
    )
    parser.add_argument(
        "--joints_root",
        default="",
        help="Directory with pre-normalization joints/ (default: <hml_root>/joints or HUMANML3D_JOINTS_ROOT)",
    )
    args = parser.parse_args()

    if int(np.__version__.split(".", maxsplit=1)[0]) >= 2:
        print("ERROR: nimblephysics requires numpy<2 for stable IK", flush=True)
        sys.exit(1)

    hml_root = Path(args.hml_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    nimble_b3d_dir(out_root).mkdir(parents=True, exist_ok=True)

    ids = all_motion_ids(hml_root)
    if args.max_motions > 0:
        ids = ids[: args.max_motions]

    _symlink_metadata(hml_root, out_root)

    joints_root = args.joints_root.strip()
    if not joints_root:
        jr = default_joints_root(hml_root)
        joints_root = str(jr) if jr is not None else ""

    work: list[WorkItem] = [
        (
            sid,
            str(hml_root),
            str(out_root),
            args.fps,
            args.mass_kg,
            args.height_m,
            args.skip_existing,
            args.joint_source,
            joints_root,
        )
        for sid in ids
    ]

    ok = err = skip = 0
    num_dofs_ref: int | None = None
    loss_sum = 0.0
    loss_count = 0

    def _record(row: dict) -> None:
        nonlocal ok, err, skip, num_dofs_ref, loss_sum, loss_count
        loss = _motion_loss(row)
        if loss is not None:
            print(f"{row['id']} Error: {loss:.6f}", flush=True)

        status = row.get("status")
        if status == "ok":
            ok += 1
            if loss is not None:
                loss_sum += loss
                loss_count += 1
            nd = int(row.get("num_dofs", 0))
            if num_dofs_ref is None:
                num_dofs_ref = nd
            elif nd != num_dofs_ref:
                print(
                    f"WARNING: num_dofs mismatch {row['id']}: {nd} vs {num_dofs_ref}",
                    flush=True,
                )
        elif status == "skipped":
            skip += 1
        else:
            err += 1

    manifest_path = out_root / "preprocess_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf:
        if args.num_workers <= 1:
            for w in work:
                row = _process_one(w)
                mf.write(json.dumps(row, default=str) + "\n")
                _record(row)
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
                futs = {ex.submit(_process_one, w): w[0] for w in work}
                for fut in as_completed(futs):
                    row = fut.result()
                    mf.write(json.dumps(row, default=str) + "\n")
                    _record(row)

    meta = {
        "hml_root": str(hml_root),
        "out_root": str(out_root),
        "nimble_b3d_dir": str(nimble_b3d_dir(out_root)),
        "fps": float(args.fps),
        "joint_source": str(args.joint_source),
        "joints_root": joints_root or None,
        "num_dofs": num_dofs_ref,
        "motions_ok": ok,
        "motions_error": err,
        "motions_skipped": skip,
    }
    (out_root / "preprocess_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("Done", flush=True)
    print(f"Successful: {ok}", flush=True)
    print(f"Unsuccessful: {err}", flush=True)
    print(f"Skipped: {skip}", flush=True)
    if loss_count > 0:
        print(f"Average Error: {loss_sum / loss_count:.6f}", flush=True)

    # Exit non-zero only if NO B3D files exist at all (no new fits + no cached
    # files from a previous run). With ``--skip_existing`` on a fully-cached
    # dataset we still want to (re)compute normalization stats.
    if ok == 0 and skip == 0:
        sys.exit(1)

    compute_nimble_normalization_stats(out_root)

    from nimble.physics import clear_cache

    clear_cache()


if __name__ == "__main__":
    main()
