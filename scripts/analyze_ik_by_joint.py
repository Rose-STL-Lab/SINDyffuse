#!/usr/bin/env python3
"""Per-joint IK error audit: HumanML3D [T,22,3] -> fit_q -> FK mismatch.

Runs against the bundled Rajagopal skeleton (the only model in
``nimble.skeleton_registry``).
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

from common.paths import default_humanml3d_root
from datasets.hml3d_joints import load_hml3d_joint_positions
from nimble.ik import _get_joint_ik_cache, fit_q
from nimble.physics import load_model
from nimble.skeleton_registry import get_spec, list_body_names, list_joint_names

HML_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

REGIONS: dict[str, list[int]] = {
    "legs_L": [1, 4, 7, 10],
    "legs_R": [2, 5, 8, 11],
    "spine_trunk": [0, 3, 6, 9, 12, 15],
    "arms_L": [13, 16, 18, 20],
    "arms_R": [14, 17, 19, 21],
}


def _proxy_position(sk: object, kind: str, name: str) -> np.ndarray:
    if kind == "joint":
        j = sk.getJoint(str(name))
        return np.asarray(sk.getJointWorldPositions([j]), dtype=np.float64).reshape(3)
    body = sk.getBodyNode(str(name))
    tr = body.getWorldTransform()
    if hasattr(tr, "translation"):
        return np.asarray(tr.translation(), dtype=np.float64).reshape(3)
    return np.asarray(tr, dtype=np.float64).reshape(3)


def _all_positions(
    sk: object,
    cache: object,
    q: np.ndarray,
    unmapped_proxies: dict[int, tuple[str, str]],
) -> dict[int, np.ndarray]:
    sk.setPositions(np.asarray(q, dtype=np.float64).reshape(-1))
    sk.computeForwardKinematics()
    flat = np.asarray(sk.getJointWorldPositions(list(cache.joints)), dtype=np.float64).reshape(
        -1, 3
    )
    out = {int(hidx): flat[k] for k, hidx in enumerate(cache.hml_indices)}
    for hidx, (kind, name) in unmapped_proxies.items():
        try:
            out[hidx] = _proxy_position(sk, kind, name)
        except Exception:
            pass
    return out


def analyze_motion(
    motion_id: str,
    root: Path,
    sk: object,
    cache: object,
    *,
    stride: int,
    ik_mapping: tuple[tuple[str, int], ...],
    unmapped_proxies: dict[int, tuple[str, str]],
) -> dict[str, float] | None:
    try:
        poses, _ = load_hml3d_joint_positions(root, motion_id, joint_source="new_joints")
    except FileNotFoundError:
        return None

    q, stats = fit_q(poses, sk, ik_mapping=ik_mapping)
    t_frames = int(poses.shape[0])
    per_hml: dict[int, list[float]] = {i: [] for i in range(22)}
    per_raj: dict[str, list[float]] = {name: [] for name in cache.joint_names}

    for t in range(0, t_frames, max(1, stride)):
        rp = _all_positions(sk, cache, q[:, t], unmapped_proxies)
        for hidx, p in rp.items():
            per_hml[hidx].append(float(np.linalg.norm(poses[t, hidx] - p)))

        targets = np.array([poses[t, j] for j in cache.hml_indices], dtype=np.float64)
        flat = np.asarray(
            sk.getJointWorldPositions(list(cache.joints)), dtype=np.float64
        ).reshape(-1, 3)
        for k, rname in enumerate(cache.joint_names):
            per_raj[rname].append(float(np.linalg.norm(flat[k] - targets[k])))

    row: dict[str, float] = {
        "motion_id": motion_id,
        "frames": float(t_frames),
        "mean_ik_error": float(stats.get("mean_ik_error", np.nan)),
        "success_ratio": float(stats.get("success_ratio", np.nan)),
    }
    for i in range(22):
        row[f"err_{HML_NAMES[i]}"] = float(np.mean(per_hml[i])) if per_hml[i] else float("nan")
    for rname in cache.joint_names:
        row[f"raj_{rname}"] = float(np.mean(per_raj[rname])) if per_raj[rname] else float("nan")
    return row


def _mean_key(rows: list[dict[str, float]], key: str) -> float:
    vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(
    rows: list[dict[str, float]],
    *,
    ik_hml_indices: frozenset,
) -> dict[str, object]:
    err_keys = [f"err_{n}" for n in HML_NAMES]
    joint_means = {k: _mean_key(rows, k) for k in err_keys}
    ranked = sorted(
        [(k.replace("err_", ""), v) for k, v in joint_means.items()],
        key=lambda x: -x[1],
    )

    region_means = {}
    for region, idxs in REGIONS.items():
        vals = [joint_means[f"err_{HML_NAMES[i]}"] for i in idxs if np.isfinite(joint_means[f"err_{HML_NAMES[i]}"])]
        region_means[region] = float(np.mean(vals)) if vals else float("nan")

    unmapped_indices = [i for i in range(22) if i not in ik_hml_indices]
    mapped_vals = [
        joint_means[f"err_{HML_NAMES[i]}"]
        for i in sorted(ik_hml_indices)
        if np.isfinite(joint_means[f"err_{HML_NAMES[i]}"])
    ]
    unmapped_vals = [
        joint_means[f"err_{HML_NAMES[i]}"]
        for i in unmapped_indices
        if np.isfinite(joint_means[f"err_{HML_NAMES[i]}"])
    ]

    raj_keys = sorted({k for r in rows for k in r if k.startswith("raj_")})
    raj_means = {k.replace("raj_", ""): _mean_key(rows, k) for k in raj_keys}
    raj_ranked = sorted(raj_means.items(), key=lambda x: -x[1])

    return {
        "n_motions": len(rows),
        "joint_ranked": ranked,
        "region_means": region_means,
        "mapped_mean_m": float(np.mean(mapped_vals)) if mapped_vals else float("nan"),
        "unmapped_mean_m": float(np.mean(unmapped_vals)) if unmapped_vals else float("nan"),
        "unmapped_joints": [HML_NAMES[i] for i in unmapped_indices],
        "raj_ranked": raj_ranked,
        "mean_ik_error": _mean_key(rows, "mean_ik_error"),
    }


def recommend(summary: dict[str, object]) -> str:
    regions = summary["region_means"]
    worst_region = max(regions.items(), key=lambda x: x[1])[0]
    unmapped = float(summary["unmapped_mean_m"])
    mapped = float(summary["mapped_mean_m"])

    return "\n".join(
        [
            f"Worst region (mean post-IK error): {worst_region} ({regions[worst_region]:.4f} m)",
            f"IK-mapped joints mean error: {mapped:.4f} m",
            f"Not in HML3D_IK mean error: {unmapped:.4f} m",
            "Unmapped HML joints (trunk/head/collars) rely on FK proxies and are "
            "not directly fit by IK.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_motions", type=int, default=80)
    parser.add_argument("--stride", type=int, default=4, help="Frame stride for FK error eval")
    parser.add_argument("--split", type=str, default="train", choices=("train", "val", "test"))
    parser.add_argument(
        "--list_joints",
        action="store_true",
        help="Just print all joint + body names of the skeleton and exit",
    )
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    spec = get_spec()
    parsed = load_model()
    sk = parsed.skeleton

    if args.list_joints:
        print(f"=== {spec.name} ({spec.description}) ===")
        print(f"ndof = {sk.getNumDofs()}")
        print("\n--- Joints ---")
        for i, n in enumerate(list_joint_names(sk)):
            print(f"  {i:3d}  {n}")
        print("\n--- Bodies ---")
        for i, n in enumerate(list_body_names(sk)):
            print(f"  {i:3d}  {n}")
        return

    root = Path(default_humanml3d_root())
    split_path = root / f"{args.split}.txt"
    if not split_path.is_file():
        raise SystemExit(f"Missing split file: {split_path}")

    ids = [ln.strip() for ln in split_path.read_text().splitlines() if ln.strip()][
        : args.max_motions
    ]

    ik_mapping = spec.ik_mapping
    cache = _get_joint_ik_cache(sk, ik_mapping=ik_mapping)
    ik_hml_indices = frozenset(int(j) for _, j in ik_mapping)
    unmapped_proxies = {int(idx): (kind, name) for idx, kind, name in spec.unmapped_proxies}

    print(
        f"Skeleton={spec.name}  ndof={sk.getNumDofs()}  "
        f"mapped HML joints={sorted(ik_hml_indices)}  "
        f"unmapped proxies={sorted(unmapped_proxies.keys())}",
        flush=True,
    )

    rows: list[dict[str, float]] = []
    skipped = 0
    for i, mid in enumerate(ids):
        row = analyze_motion(
            mid,
            root,
            sk,
            cache,
            stride=args.stride,
            ik_mapping=ik_mapping,
            unmapped_proxies=unmapped_proxies,
        )
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"processed {i + 1}/{len(ids)} ...", flush=True)

    if not rows:
        raise SystemExit("No motions analyzed (check new_joints/ paths).")

    summary = summarize(rows, ik_hml_indices=ik_hml_indices)
    print(
        f"\n=== IK audit  skeleton={spec.name}  "
        f"({summary['n_motions']} motions, stride={args.stride}) ==="
    )
    print(f"Aggregate mean_ik_error ({len(ik_mapping)} targets): {summary['mean_ik_error']:.6e}")

    print("\n--- Post-IK error by HumanML3D joint (m), worst first ---")
    for name, err in summary["joint_ranked"]:
        idx = HML_NAMES.index(name)
        if idx in ik_hml_indices:
            tag = ""
        elif idx in unmapped_proxies:
            tag = " [proxy FK, not in IK]"
        else:
            tag = " [NOT IN IK]"
        print(f"  {name:20s}  {err:.4f}{tag}")

    print("\n--- By body region (m) ---")
    for region, err in sorted(summary["region_means"].items(), key=lambda x: -x[1]):
        print(f"  {region:12s}  {err:.4f}")

    print(f"\n--- {spec.name} IK target joints (m) ---")
    for name, err in summary["raj_ranked"]:
        print(f"  {name:25s}  {err:.4f}")

    print("\n--- Recommendation ---")
    print(recommend(summary))

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "per_motion": rows}
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
