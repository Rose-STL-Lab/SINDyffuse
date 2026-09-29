"""One-time OpenSimAD muscle-tendon polynomial cache generation."""
from __future__ import annotations
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from nimble.opensimad import OPENSIM_MODEL_BASENAME
from nimble.opensimad.model_prep import ensure_ad_ready_artifacts
from nimble.opensimad.mint_settings import MINT_POLYNOMIAL_BOUNDS
from nimble.opensimad.paths import (
    ad_scaled_adjusted_model_path,
    publish_polynomial_cache,
    validate_compiled_external,
    validate_polynomial_cache,
    vendor_dummy_motion_path,
    vendor_opencap_ad_dir,
)

RIGHT_MUSCLES = (
    'addbrev_r', 'addlong_r', 'addmagDist_r', 'addmagIsch_r', 'addmagMid_r',
    'addmagProx_r', 'bflh_r', 'bfsh_r', 'edl_r', 'ehl_r', 'fdl_r', 'fhl_r',
    'gaslat_r', 'gasmed_r', 'glmax1_r', 'glmax2_r', 'glmax3_r', 'glmed1_r',
    'glmed2_r', 'glmed3_r', 'glmin1_r', 'glmin2_r', 'glmin3_r', 'grac_r',
    'iliacus_r', 'perbrev_r', 'perlong_r', 'piri_r', 'psoas_r', 'recfem_r',
    'sart_r', 'semimem_r', 'semiten_r', 'soleus_r', 'tfl_r', 'tibant_r',
    'tibpost_r', 'vasint_r', 'vaslat_r', 'vasmed_r',
)
LEFT_MUSCLES = tuple(name[:-1] + 'l' for name in RIGHT_MUSCLES)
RIGHT_JOINTS = (
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
    'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r',
)
LEFT_JOINTS = tuple(name[:-1] + 'l' for name in RIGHT_JOINTS)

def _ensure_vendor_on_path() -> None:
    vendor = vendor_opencap_ad_dir()
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))

def _write_full_range_dummy(source: Path, destination: Path) -> Path:
    """Create deterministic fitting samples spanning the supported model ROM."""
    import opensim as osim
    from utils import numpy_to_storage
    table = osim.TimeSeriesTable(str(source))
    labels = [str(label) for label in table.getColumnLabels()]
    data = np.asarray(table.getMatrix().to_numpy(), dtype=np.float64)
    times = np.asarray(table.getIndependentColumn(), dtype=np.float64).reshape(-1, 1)
    rng = np.random.default_rng(42)
    for column, label in enumerate(labels):
        coordinate = label.split('/')[-2]
        bounds = MINT_POLYNOMIAL_BOUNDS.get(coordinate)
        if bounds is None:
            continue
        values = np.linspace(float(bounds['min']), float(bounds['max']), data.shape[0])
        rng.shuffle(values)
        data[:, column] = values
    destination.parent.mkdir(parents=True, exist_ok=True)
    numpy_to_storage(['time'] + labels, np.concatenate((times, data), axis=1), str(destination), datatype='IK')
    return destination

def build_polynomial_cache(*, work_dir: Path, num_threads: int=1, force: bool=False) -> dict[str, Any]:
    """Build all model-dependent polynomial artifacts outside activation workers."""
    ensure_ad_ready_artifacts(force=False)
    validate_compiled_external(load_library=True, require_metadata=True, deep=True)
    work = Path(work_dir).expanduser().resolve()
    model_dir = work / 'Model'
    if force and model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_name = f'{OPENSIM_MODEL_BASENAME}_scaled_adjusted'
    model_path = model_dir / f'{model_name}.osim'
    shutil.copy2(ad_scaled_adjusted_model_path(), model_path)
    dummy = vendor_dummy_motion_path()
    if not dummy.is_file():
        raise FileNotFoundError(f'Missing OpenSimAD polynomial dummy motion: {dummy}')

    _ensure_vendor_on_path()
    from muscleDataOpenSimAD import getMTParameters, getPolynomialData
    full_range_dummy = _write_full_range_dummy(dummy, work / 'DummyMotionFullRange.mot')

    threads = max(1, int(num_threads))
    for side, muscles in (('r', RIGHT_MUSCLES), ('l', LEFT_MUSCLES)):
        mt_path = model_dir / f'{model_name}_mtParameters_{side}.npy'
        getMTParameters(
            str(model_path), list(muscles),
            loadMTParameters=mt_path.is_file() and not force,
            pathMTParameters=str(model_dir), modelName=model_name, side=side,
        )

    # Both sides reuse the same expensive MuscleAnalysis fitting dataset.
    for side, joints, muscles in (
        ('r', RIGHT_JOINTS, RIGHT_MUSCLES),
        ('l', LEFT_JOINTS, LEFT_MUSCLES),
    ):
        polynomial_path = model_dir / f'{model_name}_polynomial_{side}_default.npy'
        getPolynomialData(
            loadPolynomialData=polynomial_path.is_file() and not force,
            pathModelFolder=str(model_dir), modelName=model_name,
            pathMotionFile4Polynomials=str(full_range_dummy), joints=list(joints),
            muscles=list(muscles), type_bounds_polynomials='default', side=side,
            nThreads=threads, overwritedata4PolynomialFitting=bool(force and side == 'r'),
        )

    metadata_path = publish_polynomial_cache(model_dir)
    metadata = validate_polynomial_cache(require_metadata=True, deep=True)
    return {
        'metadata_path': str(metadata_path),
        'num_threads': threads,
        'artifacts': sorted(metadata['artifacts']),
    }