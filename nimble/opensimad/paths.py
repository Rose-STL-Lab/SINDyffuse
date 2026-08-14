from __future__ import annotations
from pathlib import Path
from common.paths import repo_root
from nimble.opensimad import OPENSIM_MODEL_BASENAME

def opensimad_dir() -> Path:
    return repo_root() / 'models' / 'rajagopal' / 'opensimad'

def ad_base_model_path() -> Path:
    """Unlocked + MTP-welded + function-based paths (no contacts)."""
    return opensimad_dir() / f'{OPENSIM_MODEL_BASENAME}_ad_base.osim'

def ad_contacts_model_path() -> Path:
    """AD base + foot-ground contact spheres (OpenCap naming)."""
    # OpenCap looks for: {OpenSimModel}_scaled_adjusted_contacts.osim
    return opensimad_dir() / f'{OPENSIM_MODEL_BASENAME}_scaled_adjusted_contacts.osim'

def ad_scaled_adjusted_model_path() -> Path:
    """OpenCap naming without contacts (symlink/copy of ad base)."""
    return opensimad_dir() / f'{OPENSIM_MODEL_BASENAME}_scaled_adjusted.osim'

def external_function_dir() -> Path:
    return opensimad_dir() / 'ExternalFunction'

def vendor_opencap_ad_dir() -> Path:
    return Path(__file__).resolve().parent / 'vendor' / 'opencap_ad'
