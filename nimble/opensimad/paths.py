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
    """
    dest = lai_uhlrich_dir() / 'Geometry'
    if dest.is_dir():
        return dest
    import os
    src_env = os.environ.get('LAI_GEOMETRY_SRC', '').strip()
    if src_env:
        src = Path(src_env).expanduser().resolve()
        if src.is_dir():
            if dest.is_symlink() or dest.exists():
                dest.unlink(missing_ok=True)
            try:
                dest.symlink_to(src, target_is_directory=True)
                return dest
            except OSError:
                pass
    # Soft-create empty dir so OpenSim search path exists; missing VTPs warn only.
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

def vendor_opencap_ad_dir() -> Path:
    return Path(__file__).resolve().parent / 'vendor' / 'opencap_ad'

def vendor_dummy_motion_path() -> Path:
    return Path(__file__).resolve().parent / 'vendor' / 'opencap_pipeline' / 'MuscleAnalysis' / 'DummyMotion.mot'
