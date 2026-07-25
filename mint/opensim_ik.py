"""OpenSim Scale + IK for MinT SSM67 marker trajectories."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np

from mint.coord_map import default_mint_model_path, mint_coordinate_names
from mint.ssm67_markers import ensure_ssm67_ik_setup_xml, ensure_ssm67_marker_xml


def _prepare_model_with_markers(model_path: Path, marker_xml: Path, out_path: Path) -> Path:
    import opensim as osim

    model = osim.Model(str(model_path))
    marker_set = osim.MarkerSet(str(marker_xml))
    model.set_MarkerSet(marker_set)
    model.finalizeConnections()
    model.initSystem()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.printToXML(str(out_path))
    return out_path


def _patch_setup_xml(src: Path, dst: Path, replacements: dict[str, str]) -> None:
    text = src.read_text(encoding="utf-8")
    for key, val in replacements.items():
        text = text.replace(f"<{key}>Unassigned</{key}>", f"<{key}>{val}</{key}>")
    dst.write_text(text, encoding="utf-8")


def _read_mot_q(mot_path: Path, coord_names: List[str]) -> np.ndarray:
    import opensim as osim

    table = osim.TimeSeriesTable(str(mot_path))
    times = np.array(table.getIndependentColumn(), dtype=np.float64)
    if len(times) == 0:
        raise RuntimeError(f"Empty motion file: {mot_path}")
    labels = list(table.getColumnLabels())
    label_to_idx = {str(l): i for i, l in enumerate(labels)}
    rows = []
    mat = table.getMatrix().to_numpy()
    for name in coord_names:
        if name not in label_to_idx:
            raise KeyError(f"Coordinate {name!r} missing from {mot_path}; have {labels[:8]}...")
        rows.append(mat[:, label_to_idx[name]])
    q = np.stack(rows, axis=-1).astype(np.float32)
    return q


def run_opensim_ik_from_trc(
    trc_path: str | Path,
    *,
    fps: float = 20.0,
    work_dir: str | Path | None = None,
) -> Tuple[np.ndarray, float]:
    """Run OpenSim IK on ``trc_path``; return ``q`` ``[T, ndof]`` and mean marker error (m)."""
    import opensim as osim

    trc_path = Path(trc_path).resolve()
    model_path = default_mint_model_path()
    assets = model_path.parent / "opensim"
    marker_xml = ensure_ssm67_marker_xml()
    ik_setup_src = ensure_ssm67_ik_setup_xml()
    coord_names = list(mint_coordinate_names())

    cleanup = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="mint_ik_"))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        model_with_markers = _prepare_model_with_markers(
            model_path,
            marker_xml,
            work_dir / "LaiUhlrich2022_ssm67.osim",
        )
        ik_setup = work_dir / "Setup_IK_ssm67_resolved.xml"
        mot_out = work_dir / "ik.mot"
        _patch_setup_xml(
            ik_setup_src,
            ik_setup,
            {
                "model_file": str(model_with_markers),
                "marker_file": str(trc_path),
                "output_motion_file": str(mot_out),
                "results_directory": str(work_dir),
            },
        )
        ik = osim.InverseKinematicsTool(str(ik_setup))
        ik.run()
        if not mot_out.is_file():
            raise RuntimeError(f"OpenSim IK did not produce {mot_out}")
        q = _read_mot_q(mot_out, coord_names)

        mean_err = 0.0
        err_path = work_dir / "ik_marker_errors.sto"
        if err_path.is_file():
            err_table = osim.TimeSeriesTable(str(err_path))
            err_mat = err_table.getMatrix().to_numpy()
            if err_mat.size > 0:
                mean_err = float(np.sqrt(np.mean(err_mat**2)))
        return q, mean_err
    finally:
        if cleanup and os.environ.get("MINT_IK_KEEP_WORKDIR", "").strip() != "1":
            shutil.rmtree(work_dir, ignore_errors=True)


def run_mint_opensim_pipeline(
    marker_positions: np.ndarray,
    *,
    fps: float = 20.0,
    work_dir: str | Path | None = None,
) -> Tuple[np.ndarray, float]:
    """Write TRC from ``marker_positions`` ``[T,67,3]`` and run IK."""
    from mint.trc_io import write_trc

    cleanup = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="mint_retarget_"))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    try:
        trc_path = work_dir / "markers.trc"
        write_trc(trc_path, marker_positions, fps=fps)
        return run_opensim_ik_from_trc(trc_path, fps=fps, work_dir=work_dir)
    finally:
        if cleanup and os.environ.get("MINT_IK_KEEP_WORKDIR", "").strip() != "1":
            shutil.rmtree(work_dir, ignore_errors=True)
