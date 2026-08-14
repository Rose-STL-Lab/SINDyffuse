from __future__ import annotations
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
from nimble.moco_segment import SIM_GRF_COLS
from nimble.muscle_activation import MuscleActivationConfig, muscle_names
from nimble.opensimad import OPENSIM_MODEL_BASENAME
from nimble.opensimad.mint_settings import mint_tracking_settings
from nimble.opensimad.model_prep import ensure_ad_ready_artifacts
from nimble.opensimad.paths import ad_contacts_model_path, ad_scaled_adjusted_model_path, external_function_dir, vendor_opencap_ad_dir
from nimble.rajagopal_coord_map import build_rajagopal_coord_mapping, write_coordinates_mot

def _ensure_vendor_on_path() -> None:
    vendor = vendor_opencap_ad_dir()
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))

def _parse_activations_mot(path: Path, *, n_frames: int, muscle_name_list: Tuple[str, ...], fps: float) -> np.ndarray:
    import opensim as osim
    table = osim.TimeSeriesTable(str(path))
    labels = list(table.getColumnLabels())
    times = np.array([float(t) for t in table.getIndependentColumn()], dtype=np.float64)
    data = table.getMatrix().to_numpy()
    frame_times = np.arange(n_frames, dtype=np.float64) / max(float(fps), 1e-8)
    out = np.full((n_frames, len(muscle_name_list)), np.nan, dtype=np.float32)
    label_to_col = {str(lab): i for i, lab in enumerate(labels)}
    for mi, name in enumerate(muscle_name_list):
        col = None
        for lab, idx in label_to_col.items():
            if lab == name or lab.endswith(f'/{name}') or f'/{name}/' in lab or lab.endswith(f'|{name}'):
                col = idx
                break
            # OpenCap often uses bare muscle names in activation columns.
            if name in lab and 'activation' in lab.lower():
                col = idx
                break
        if col is None:
            # try exact muscle name as substring at end
            for lab, idx in label_to_col.items():
                if lab.split('/')[-1] == name or lab.split('|')[-1] == name:
                    col = idx
                    break
        if col is None:
            continue
        series = data[:, col]
        finite = np.isfinite(times) & np.isfinite(series)
        if finite.sum() < 2:
            continue
        out[:, mi] = np.interp(frame_times, times[finite], series[finite], left=np.nan, right=np.nan).astype(np.float32)
    return out

def _parse_grf_mot(path: Path | None, *, n_frames: int, fps: float) -> np.ndarray:
    grf = np.full((n_frames, SIM_GRF_COLS), np.nan, dtype=np.float32)
    if path is None or not path.is_file():
        return grf
    import opensim as osim
    table = osim.TimeSeriesTable(str(path))
    labels = [str(x) for x in table.getColumnLabels()]
    times = np.array([float(t) for t in table.getIndependentColumn()], dtype=np.float64)
    data = table.getMatrix().to_numpy()
    frame_times = np.arange(n_frames, dtype=np.float64) / max(float(fps), 1e-8)
    # Best-effort map of common OpenCap GRF labels into our 18-col pack (left/right force+torque).
    wanted = [
        'ground_force_left_vx', 'ground_force_left_vy', 'ground_force_left_vz',
        'ground_torque_left_x', 'ground_torque_left_y', 'ground_torque_left_z',
        'ground_force_right_vx', 'ground_force_right_vy', 'ground_force_right_vz',
        'ground_torque_right_x', 'ground_torque_right_y', 'ground_torque_right_z',
    ]
    for i, key in enumerate(wanted):
        col = None
        for li, lab in enumerate(labels):
            if key in lab or lab.endswith(key):
                col = li
                break
        if col is None:
            continue
        series = data[:, col]
        finite = np.isfinite(times) & np.isfinite(series)
        if finite.sum() < 2:
            continue
        grf[:, i] = np.interp(frame_times, times[finite], series[finite], left=np.nan, right=np.nan).astype(np.float32)
    # Derived channels
    if np.isfinite(grf[:, 1]).any() and np.isfinite(grf[:, 7]).any():
        grf[:, 12] = grf[:, 1] + grf[:, 7]  # vertical
        grf[:, 13] = np.linalg.norm(grf[:, 0:3], axis=1)
        grf[:, 14] = np.linalg.norm(grf[:, 6:9], axis=1)
        grf[:, 15] = np.linalg.norm(grf[:, 3:6], axis=1)
        grf[:, 16] = np.linalg.norm(grf[:, 9:12], axis=1)
        grf[:, 17] = 1.0
    return grf

def solve_opensimad_segment(q: np.ndarray, *, cfg: MuscleActivationConfig, solve_dir: Path, mesh_interval: float | None=None) -> Tuple[np.ndarray, bool, Dict[str, Any], np.ndarray]:
    """Run one MinT/OpenCap OpenSimAD tracking window on Rajagopal coordinates."""
    ensure_ad_ready_artifacts(force=False)
    ext_dir = external_function_dir()
    if not (ext_dir / 'F_map.npy').is_file():
        raise FileNotFoundError(
            f'Missing OpenSimAD artifacts under {ext_dir}. '
            'Run: python scripts/build_rajagopal_opensimad_ext.py'
        )
    _ensure_vendor_on_path()
    arr = np.asarray(q, dtype=np.float64)
    n_frames = int(arr.shape[0])
    names = muscle_names()
    solve_dir = Path(solve_dir)
    solve_dir.mkdir(parents=True, exist_ok=True)

    # OpenCap-like session layout for run_tracking.
    subject = 'sindyffuse'
    data_dir = solve_dir / 'data'
    session = data_dir / subject
    model_folder = session / 'OpenSimData' / 'Model'
    kin_folder = session / 'OpenSimData' / 'Kinematics'
    dyn_folder = session / 'OpenSimData' / 'Dynamics' / 'segment'
    model_folder.mkdir(parents=True, exist_ok=True)
    kin_folder.mkdir(parents=True, exist_ok=True)
    dyn_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ad_scaled_adjusted_model_path(), model_folder / f'{OPENSIM_MODEL_BASENAME}_scaled_adjusted.osim')
    shutil.copy2(ad_contacts_model_path(), model_folder / f'{OPENSIM_MODEL_BASENAME}_scaled_adjusted_contacts.osim')
    ext_dst = model_folder / 'ExternalFunction'
    if ext_dst.exists():
        shutil.rmtree(ext_dst)
    shutil.copytree(ext_dir, ext_dst)

    mapping = build_rajagopal_coord_mapping(model_path=ad_scaled_adjusted_model_path())
    mot_path = kin_folder / 'segment.mot'
    write_coordinates_mot(arr, mot_path, fps=float(cfg.fps), mapping=mapping)
    (session / 'sessionMetadata.yaml').write_text(
        f"openSimModel: {OPENSIM_MODEL_BASENAME}\nmass_kg: {float(cfg.mass_kg)}\nheight_m: 1.75\n",
        encoding='utf-8',
    )

    t1 = max((n_frames - 1) / max(float(cfg.fps), 1e-8), 1e-3)
    settings = mint_tracking_settings(mass_kg=float(cfg.mass_kg), trial_name='segment')
    settings['timeInterval'] = [0.0, float(t1)]
    if mesh_interval is not None and float(mesh_interval) > 0:
        settings['meshDensity'] = int(round(1.0 / float(mesh_interval)))

    # Fake OpenCap baseDir so imports + opensimAD-install resolve under vendor.
    fake_base = solve_dir / 'opencap_base'
    link = fake_base / 'UtilsDynamicSimulations' / 'OpenSimAD'
    link.parent.mkdir(parents=True, exist_ok=True)
    vendor = vendor_opencap_ad_dir()
    if link.exists() or link.is_symlink():
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            shutil.rmtree(link, ignore_errors=True)
    try:
        link.symlink_to(vendor, target_is_directory=True)
    except OSError:
        shutil.copytree(vendor, link, dirs_exist_ok=True)

    meta: Dict[str, Any] = {
        'activation_method': 'opensimad',
        'mesh_interval': float(mesh_interval) if mesh_interval else 1.0 / float(settings['meshDensity']),
        'max_iterations': int(settings['max_iterations']),
    }
    try:
        from mainOpenSimAD import run_tracking
        run_tracking(
            str(fake_base),
            str(data_dir),
            subject,
            settings,
            case='0',
            solveProblem=True,
            analyzeResults=True,
            writeGUI=False,
            computeKAM=False,
            computeMCF=False,
        )
        act_mot = next(dyn_folder.glob('kinematics_activations_*.mot'), None)
        grf_mot = next(dyn_folder.glob('GRF_*.mot'), None)
        if act_mot is None:
            # OpenCap may nest under trial subfolder
            act_mot = next(session.joinpath('OpenSimData', 'Dynamics').rglob('kinematics_activations_*.mot'), None)
            grf_mot = next(session.joinpath('OpenSimData', 'Dynamics').rglob('GRF_*.mot'), None)
        if act_mot is None:
            raise RuntimeError('OpenSimAD finished without kinematics_activations_*.mot')
        activations = _parse_activations_mot(act_mot, n_frames=n_frames, muscle_name_list=names, fps=float(cfg.fps))
        grf = _parse_grf_mot(grf_mot, n_frames=n_frames, fps=float(cfg.fps))
        ok = bool(np.isfinite(activations).any())
        meta['solver_success'] = ok
        meta['solver_status'] = 'ok' if ok else 'no_finite_activations'
        return (activations, ok, meta, grf)
    except Exception as exc:
        meta['solver_success'] = False
        meta['solver_status'] = 'error'
        meta['error'] = str(exc)
        activations = np.full((n_frames, len(names)), np.nan, dtype=np.float32)
        grf = np.full((n_frames, SIM_GRF_COLS), np.nan, dtype=np.float32)
        return (activations, False, meta, grf)
