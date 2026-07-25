#!/usr/bin/env python3
"""Preprocess HumanML3D → MinT OpenSim q + MinT muscle labels (NPZ cache).

Example::

  python scripts/preprocess_mint.py --max_motions 1
  python scripts/preprocess_mint.py --mint_root /path/to/MinT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.paths import default_humanml3d_root, default_mint_root, mint_cache_dir
from common.run_logging import RunLogger, add_run_log_cli_args, dual_tqdm, null_logger, run_logged_main
from common.skeleton_config import DEFAULT_FPS
from datasets.hml3d_joints import default_joints_root, load_hml3d_joint_positions
from datasets.mint_cache_stats import compute_mint_normalization_stats
from datasets.splits import all_motion_ids, shard_motion_ids
from mint.cache_schema import write_cache_metadata, write_motion_cache
from mint.features import features_from_mint_q
from mint.label_lookup import lookup_hml_motion
from mint.physics import bio_matrix_mint
from mint.retarget import retarget_hml_joints_to_q


def export_motion_to_npz(
    motion_id: str,
    *,
    hml_root: Path,
    out_root: Path,
    mint_root: str,
    fps: float = DEFAULT_FPS,
    skip_existing: bool = False,
    joint_source: str = "auto",
    joints_root: str = "",
) -> dict:
    out_path = mint_cache_dir(out_root) / f"{motion_id}.npz"
    if skip_existing and out_path.is_file():
        return {"id": motion_id, "status": "skipped", "path": str(out_path)}

    try:
        joints = load_hml3d_joint_positions(
            motion_id,
            hml_root=str(hml_root),
            joint_source=joint_source,
            joints_root=joints_root or None,
        )
        rt = retarget_hml_joints_to_q(joints, fps=fps)
        q = rt.q
        t_len = int(q.shape[0])

        label = lookup_hml_motion(
            motion_id,
            mint_root=mint_root,
            num_frames=t_len,
            target_fps=fps,
        )
        act = label.activations
        if act.shape[0] != t_len:
            if act.shape[0] > t_len:
                act = act[:t_len]
            else:
                pad = np.zeros((t_len - act.shape[0], act.shape[1]), dtype=np.float32)
                act = np.concatenate([act, pad], axis=0)

        bio = bio_matrix_mint(q, fps=fps)
        u, c, _, _ = features_from_mint_q(q, fps=fps)
        sindy_feat = np.concatenate([u, c], axis=-1).astype(np.float32)

        write_motion_cache(
            out_path,
            q=q,
            muscle_activations=act,
            guidance_features=bio,
            sindy_features=sindy_feat,
            has_mint_labels=label.has_labels,
            fps=fps,
        )
        return {
            "id": motion_id,
            "status": "ok",
            "path": str(out_path),
            "num_frames": t_len,
            "num_dofs": int(q.shape[1]),
            "has_mint_labels": bool(label.has_labels),
            "mean_fk_error": float(rt.mean_fk_error),
        }
    except Exception as exc:
        return {"id": motion_id, "status": "error", "error": str(exc)}


def run_preprocess(args: argparse.Namespace, logger: RunLogger | None = None) -> None:
    log = logger or null_logger()
    hml_root = Path(args.hml_root or default_humanml3d_root()).expanduser().resolve()
    out_root = Path(args.out_root or default_humanml3d_root()).expanduser().resolve()
    mint_root = str(args.mint_root or default_mint_root())
    cache = mint_cache_dir(out_root)
    cache.mkdir(parents=True, exist_ok=True)

    ids = all_motion_ids(hml_root)
    if int(args.max_motions) > 0:
        ids = ids[: int(args.max_motions)]

    shard_index = int(getattr(args, "shard_index", 0) or 0)
    num_shards = int(getattr(args, "num_shards", 1) or 1)
    if num_shards > 1:
        ids = shard_motion_ids(ids, shard_index, num_shards)
        log.progress(f"Shard {shard_index + 1}/{num_shards}: {len(ids)} motion(s)")

    joints_root = str(args.joints_root or "").strip()
    if not joints_root:
        jr = default_joints_root(hml_root)
        joints_root = str(jr) if jr is not None else ""

    fps = float(args.fps)
    ok = err = skip = 0
    num_dofs = None

    with dual_tqdm(total=len(ids), desc="preprocess_mint", logger=log) as pbar:
        for sid in ids:
            row = export_motion_to_npz(
                sid,
                hml_root=hml_root,
                out_root=out_root,
                mint_root=mint_root,
                fps=fps,
                skip_existing=bool(args.skip_existing),
                joint_source=str(args.joint_source),
                joints_root=joints_root,
            )
            st = row.get("status")
            if st == "ok":
                ok += 1
                nd = int(row.get("num_dofs", 0))
                num_dofs = nd if num_dofs is None else num_dofs
            elif st == "skipped":
                skip += 1
            else:
                err += 1
                log.verbose(f"ERROR {sid}: {row.get('error')}")
            pbar.update(1)

    log.progress(f"Done: ok={ok} skip={skip} err={err}")
    if num_dofs is not None:
        write_cache_metadata(cache, ndof=int(num_dofs), fps=fps)

    if not getattr(args, "skip_normalization", False) and ok > 0:
        stats = compute_mint_normalization_stats(out_root, split="train")
        log.progress(f"Wrote normalization: {stats}")

    manifest = cache / f"preprocess_mint_manifest.{shard_index:04d}.jsonl"
    manifest.write_text(
        json.dumps({"ok": ok, "skip": skip, "err": err, "num_dofs": num_dofs}) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess HumanML3D into MinT NPZ cache")
    parser.add_argument("--hml_root", default="", help="HumanML3D root")
    parser.add_argument("--out_root", default="", help="Output root (default: hml_root)")
    parser.add_argument("--mint_root", default="", help="MinT dataset root (MINT_ROOT)")
    parser.add_argument("--max_motions", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--skip_normalization", action="store_true")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--joint_source", default="auto")
    parser.add_argument("--joints_root", default="")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    add_run_log_cli_args(parser)
    args = parser.parse_args()

    run_logged_main(
        Path(__file__).stem,
        args.log_dir,
        lambda logger: run_preprocess(args, logger),
        argv=sys.argv,
        no_run_log=bool(args.no_run_log),
        run_id=str(getattr(args, "run_log_id", "") or "").strip() or None,
    )


if __name__ == "__main__":
    main()
