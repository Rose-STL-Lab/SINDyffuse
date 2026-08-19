from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path

# CasADi before OpenSim (see scripts/build_rajagopal_opensimad_ext.py).
import casadi  # noqa: F401

from nimble.opensimad.model_prep import ensure_ad_ready_artifacts
from nimble.opensimad.paths import ad_contacts_model_path, ad_scaled_adjusted_model_path, external_function_dir, opensimad_dir, vendor_opencap_ad_dir
from nimble.opensimad import OPENSIM_MODEL_BASENAME

def _ensure_vendor_on_path() -> Path:
    vendor = vendor_opencap_ad_dir()
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    return vendor

def build_rajagopal_opensimad_external(*, force: bool=False, use_expression_graph: bool=True) -> Path:
    """Generate OpenSimAD external function F for the AD-ready Rajagopal contacts model.

    Downloads OpenCap's opensimAD-install toolchain on first run (Linux/macOS/Windows).
    Artifacts land in models/rajagopal/opensimad/ExternalFunction/.
    """
    ensure_ad_ready_artifacts(force=force)
    out_dir = external_function_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    ext_py = out_dir / 'F.py'
    ext_map = out_dir / 'F_map.npy'
    if not force and ext_py.is_file() and ext_map.is_file():
        return out_dir

    vendor = _ensure_vendor_on_path()
    # Layout expected by OpenCap generateExternalFunction:
    #   dataDir/subject/OpenSimData/Model/{Model}_scaled_adjusted_contacts.osim
    staging_root = opensimad_dir() / '_codegen_staging'
    subject = 'sindyffuse'
    model_folder = staging_root / subject / 'OpenSimData' / 'Model'
    if staging_root.exists() and force:
        shutil.rmtree(staging_root, ignore_errors=True)
    model_folder.mkdir(parents=True, exist_ok=True)
    contacts = ad_contacts_model_path()
    scaled = ad_scaled_adjusted_model_path()
    shutil.copy2(scaled, model_folder / f'{OPENSIM_MODEL_BASENAME}_scaled_adjusted.osim')
    shutil.copy2(contacts, model_folder / f'{OPENSIM_MODEL_BASENAME}_scaled_adjusted_contacts.osim')

    from utilsOpenSimAD import generateExternalFunction
    # baseDir must contain UtilsDynamicSimulations/OpenSimAD OR we pass vendor as pathDCAD via monkeypatch.
    # generateExternalFunction uses baseDir/UtilsDynamicSimulations/OpenSimAD for build — emulate that layout.
    fake_base = opensimad_dir() / '_opencap_base'
    ad_link = fake_base / 'UtilsDynamicSimulations' / 'OpenSimAD'
    ad_link.parent.mkdir(parents=True, exist_ok=True)
    if ad_link.exists() or ad_link.is_symlink():
        if ad_link.is_symlink() or ad_link.is_file():
            ad_link.unlink()
        else:
            shutil.rmtree(ad_link, ignore_errors=True)
    try:
        ad_link.symlink_to(vendor, target_is_directory=True)
    except OSError:
        shutil.copytree(vendor, ad_link, dirs_exist_ok=True)

    generateExternalFunction(
        str(fake_base),
        str(staging_root),
        subject,
        OpenSimModel=OPENSIM_MODEL_BASENAME,
        treadmill=False,
        build_externalFunction=True,
        verifyID=False,
        externalFunctionName='F',
        overwrite=bool(force),
        useExpressionGraphFunction=bool(use_expression_graph),
        contact_side='all',
        trial_name=None,
    )
    staged_ext = model_folder / 'ExternalFunction'
    if staged_ext.is_dir():
        for p in staged_ext.iterdir():
            dest = out_dir / p.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(p), str(dest))
    if not (out_dir / 'F_map.npy').is_file():
        raise RuntimeError(f'OpenSimAD codegen failed; missing F_map.npy under {out_dir}')
    return out_dir
