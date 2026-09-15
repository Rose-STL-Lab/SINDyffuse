from __future__ import annotations
from pathlib import Path
from common.paths import repo_root
from nimble.opensimad import OPENSIM_MODEL_BASENAME

def lai_uhlrich_dir() -> Path:
    return repo_root() / 'models' / 'lai_uhlrich'

def ensure_lai_geometry() -> Path:
    """Ensure ``models/lai_uhlrich/Geometry`` exists for OpenSim mesh lookup.

    Prefer a local Geometry directory next to the model. Optional fallback:
    symlink into an existing OpenSim/OpenCap Geometry tree if LAI_GEOMETRY_SRC
    is set. Mesh files are not required for IK/OpenSimAD solves.

    Safe under concurrent K8s pods: ignore races on create, and replace a
    broken leftover symlink (e.g. old nimblephysics Geometry path).
    """
    dest = lai_uhlrich_dir() / 'Geometry'
    if dest.is_dir():
        return dest
    # Broken symlink: exists on disk but does not resolve to a directory.
    if dest.is_symlink() or dest.exists():
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            if dest.is_dir():
                return dest
            raise
    import os
    src_env = os.environ.get('LAI_GEOMETRY_SRC', '').strip()
    if src_env:
        src = Path(src_env).expanduser().resolve()
        if src.is_dir():
            try:
                dest.symlink_to(src, target_is_directory=True)
                return dest
            except FileExistsError:
                if dest.is_dir():
                    return dest
            except OSError:
                pass
    # Soft-create empty dir so OpenSim search path exists; missing VTPs warn only.
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        # Another pod won the race (dir or symlink). Accept if usable.
        if dest.is_dir():
            return dest
        if dest.is_symlink() or dest.exists():
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
            dest.mkdir(parents=True, exist_ok=True)
    return dest

def lai_uhlrich_model_path() -> Path:
    ensure_lai_geometry()
    return lai_uhlrich_dir() / f'{OPENSIM_MODEL_BASENAME}.osim'

def opensimad_dir() -> Path:
    return lai_uhlrich_dir() / 'opensimad'

def ad_base_model_path() -> Path:
    """Unlocked + MTP-welded AD base (no contacts)."""
    return opensimad_dir() / f'{OPENSIM_MODEL_BASENAME}_ad_base.osim'

def ad_contacts_model_path() -> Path:
    """AD base + foot-ground contact spheres (OpenCap naming)."""
    return opensimad_dir() / f'{OPENSIM_MODEL_BASENAME}_scaled_adjusted_contacts.osim'

def ad_scaled_adjusted_model_path() -> Path:
    """OpenCap naming without contacts (copy of AD base)."""
    return opensimad_dir() / f'{OPENSIM_MODEL_BASENAME}_scaled_adjusted.osim'

def external_function_dir() -> Path:
    return opensimad_dir() / 'ExternalFunction'

def polynomial_cache_globs() -> tuple[str, ...]:
    """OpenCap MuscleAnalysis / polynomial artifacts keyed by scaled model name."""
    base = f'{OPENSIM_MODEL_BASENAME}_scaled_adjusted'
    return (
        f'{base}_polynomial_*.npy',
        f'{base}_mtParameters_*.npy',
        f'data4PolynomialFitting_{base}_*.npy',
    )

def stage_opensimad_polynomial_cache(model_folder: Path) -> int:
    """Copy shared polynomial / MT-parameter caches into a per-segment Model dir.

    Without this, every segment re-runs MuscleAnalysis (joblib×host CPUs) and OOMs.
    """
    import shutil
    src = opensimad_dir()
    model_folder = Path(model_folder)
    model_folder.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pattern in polynomial_cache_globs():
        for path in src.glob(pattern):
            if not path.is_file():
                continue
            dest = model_folder / path.name
            try:
                shutil.copy2(path, dest)
                copied += 1
            except OSError:
                pass
    return copied

def promote_opensimad_polynomial_cache(model_folder: Path) -> int:
    """Publish newly fitted polynomial caches back to the shared opensimad dir."""
    import os
    import shutil
    import tempfile
    src_dir = Path(model_folder)
    dst_dir = opensimad_dir()
    dst_dir.mkdir(parents=True, exist_ok=True)
    promoted = 0
    for pattern in polynomial_cache_globs():
        for path in src_dir.glob(pattern):
            if not path.is_file():
                continue
            dest = dst_dir / path.name
            if dest.is_file() and dest.stat().st_size > 0:
                continue
            try:
                fd, tmp_name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=str(dst_dir))
                os.close(fd)
                tmp_path = Path(tmp_name)
                try:
                    shutil.copy2(path, tmp_path)
                    os.replace(tmp_path, dest)
                    promoted += 1
                except OSError:
                    tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
    return promoted

def vendor_opencap_ad_dir() -> Path:
    return Path(__file__).resolve().parent / 'vendor' / 'opencap_ad'

def vendor_dummy_motion_path() -> Path:
    return Path(__file__).resolve().parent / 'vendor' / 'opencap_pipeline' / 'MuscleAnalysis' / 'DummyMotion.mot'
