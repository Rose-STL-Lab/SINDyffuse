from __future__ import annotations
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from common.paths import repo_root
from nimble.opensimad import OPENSIM_MODEL_BASENAME

EXTERNAL_FUNCTION_METADATA = 'compiled_external.json'
POLYNOMIAL_CACHE_METADATA = 'polynomial_cache.json'
POLYNOMIAL_PIPELINE_VERSION = 'degrees_and_descending_model_coefficients_v1'

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

def compiled_external_suffix() -> str:
    system = platform.system()
    if system == 'Linux':
        return '.so'
    if system == 'Darwin':
        return '.dylib'
    if system == 'Windows':
        return '.dll'
    raise RuntimeError(f'Unsupported OpenSimAD platform: {system}')

def compiled_external_path() -> Path:
    return external_function_dir() / f'F{compiled_external_suffix()}'

def external_function_metadata_path() -> Path:
    return external_function_dir() / EXTERNAL_FUNCTION_METADATA

def polynomial_cache_metadata_path() -> Path:
    return opensimad_dir() / POLYNOMIAL_CACHE_METADATA

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'Invalid OpenSimAD metadata file {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise RuntimeError(f'Invalid OpenSimAD metadata object in {path}')
    return value

def write_external_function_metadata() -> Path:
    import casadi as ca
    library = compiled_external_path()
    map_path = external_function_dir() / 'F_map.npy'
    contacts = ad_contacts_model_path()
    payload = {
        'artifact_type': 'opensimad_compiled_external',
        'model': OPENSIM_MODEL_BASENAME,
        'model_sha256': _sha256(contacts),
        'library': library.name,
        'library_sha256': _sha256(library),
        'map_sha256': _sha256(map_path),
        'casadi_version': str(ca.__version__),
        'platform': platform.system(),
        'machine': platform.machine(),
        'complete': True,
    }
    path = external_function_metadata_path()
    _write_json_atomic(path, payload)
    return path

def validate_compiled_external(*, load_library: bool=True, require_metadata: bool=True, deep: bool=True) -> dict:
    import numpy as np
    library = compiled_external_path()
    map_path = external_function_dir() / 'F_map.npy'
    missing = [str(path) for path in (library, map_path) if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(
            'Missing compiled OpenSimAD external-function artifact(s): '
            + ', '.join(missing)
            + '. Run scripts/build_lai_opensimad_ext.py --force.'
        )
    try:
        fmap = np.load(map_path, allow_pickle=True).item()
    except Exception as exc:
        raise RuntimeError(f'Could not load OpenSimAD map {map_path}: {exc}') from exc
    if not isinstance(fmap, dict) or not {'residuals', 'GRFs'}.issubset(fmap):
        raise RuntimeError(f'OpenSimAD map {map_path} is missing residuals/GRFs metadata')
    if load_library:
        import casadi as ca
        try:
            external = ca.external('F', str(library))
            if int(external.n_in()) != 1 or int(external.n_out()) != 1:
                raise RuntimeError(
                    f'unexpected CasADi signature n_in={external.n_in()} n_out={external.n_out()}'
                )
        except Exception as exc:
            raise RuntimeError(f'Could not load compiled OpenSimAD external {library}: {exc}') from exc
    if not require_metadata:
        return {'library': str(library), 'map': str(map_path)}
    metadata_path = external_function_metadata_path()
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f'Missing compiled OpenSimAD metadata {metadata_path}. '
            'Re-run scripts/build_lai_opensimad_ext.py --force.'
        )
    metadata = _load_json(metadata_path)
    import casadi as ca
    expected = {
        'artifact_type': 'opensimad_compiled_external',
        'model': OPENSIM_MODEL_BASENAME,
        'model_sha256': _sha256(ad_contacts_model_path()),
        'library': library.name,
        'casadi_version': str(ca.__version__),
        'platform': platform.system(),
        'machine': platform.machine(),
        'complete': True,
    }
    if deep:
        expected['library_sha256'] = _sha256(library)
        expected['map_sha256'] = _sha256(map_path)
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise RuntimeError(
            f'Stale or incompatible compiled OpenSimAD external metadata ({", ".join(mismatches)}). '
            'Re-run scripts/build_lai_opensimad_ext.py --force.'
        )
    return metadata

def required_polynomial_cache_names() -> tuple[str, ...]:
    base = f'{OPENSIM_MODEL_BASENAME}_scaled_adjusted'
    return (
        f'{base}_mtParameters_l.npy',
        f'{base}_mtParameters_r.npy',
        f'{base}_polynomial_l_default.npy',
        f'{base}_polynomial_r_default.npy',
        f'data4PolynomialFitting_{base}_default.npy',
    )

def runtime_polynomial_cache_names() -> tuple[str, ...]:
    return tuple(name for name in required_polynomial_cache_names() if not name.startswith('data4PolynomialFitting_'))

def _validate_polynomial_files(paths: list[Path]) -> None:
    import numpy as np
    try:
        mt_l = np.load(paths[0], allow_pickle=True)
        mt_r = np.load(paths[1], allow_pickle=True)
        poly_l = np.load(paths[2], allow_pickle=True).item()
        poly_r = np.load(paths[3], allow_pickle=True).item()
        fitting = np.load(paths[4], allow_pickle=True).item()
    except Exception as exc:
        raise RuntimeError(f'Could not load OpenSimAD polynomial cache: {exc}') from exc
    if mt_l.shape[0] != 5 or mt_r.shape[0] != 5 or mt_l.shape[1] == 0 or mt_r.shape[1] == 0:
        raise RuntimeError(f'Invalid muscle-tendon parameter shapes: left={mt_l.shape}, right={mt_r.shape}')
    if not isinstance(poly_l, dict) or not poly_l or not isinstance(poly_r, dict) or not poly_r:
        raise RuntimeError('OpenSimAD left/right polynomial dictionaries must be non-empty')
    required_fitting = {'mtu_lengths', 'mtu_moment_arms', 'muscle_names', 'coordinate_names', 'coordinate_values'}
    if not isinstance(fitting, dict) or not required_fitting.issubset(fitting):
        raise RuntimeError('OpenSimAD polynomial fitting data is missing required arrays')

def validate_polynomial_cache(*, require_metadata: bool=True, deep: bool=True) -> dict:
    import numpy as np
    root = opensimad_dir()
    paths = [root / name for name in required_polynomial_cache_names()]
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(
            'Missing precomputed OpenSimAD polynomial cache artifact(s): '
            + ', '.join(missing)
            + '. Run scripts/build_lai_opensimad_polynomials.py --force.'
        )
    if deep:
        _validate_polynomial_files(paths)
    if not require_metadata:
        return {'artifacts': [str(path) for path in paths]}
    metadata_path = polynomial_cache_metadata_path()
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f'Missing polynomial cache metadata {metadata_path}. '
            'Re-run scripts/build_lai_opensimad_polynomials.py --force.'
        )
    metadata = _load_json(metadata_path)
    expected = {
        'artifact_type': 'opensimad_polynomial_cache',
        'pipeline_version': POLYNOMIAL_PIPELINE_VERSION,
        'model': OPENSIM_MODEL_BASENAME,
        'model_sha256': _sha256(ad_scaled_adjusted_model_path()),
        'complete': True,
    }
    from nimble.opensimad.mint_settings import MINT_POLYNOMIAL_BOUNDS
    expected['polynomial_bounds_degrees'] = MINT_POLYNOMIAL_BOUNDS
    if deep:
        expected['artifacts'] = {path.name: _sha256(path) for path in paths}
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise RuntimeError(
            f'Stale or incomplete OpenSimAD polynomial metadata ({", ".join(mismatches)}). '
            'Re-run scripts/build_lai_opensimad_polynomials.py --force.'
        )
    return metadata

def publish_polynomial_cache(source_dir: Path) -> Path:
    import shutil
    source = Path(source_dir)
    destination = opensimad_dir()
    destination.mkdir(parents=True, exist_ok=True)
    names = required_polynomial_cache_names()
    for name in names:
        src = source / name
        if not src.is_file() or src.stat().st_size <= 0:
            raise FileNotFoundError(f'Polynomial cache build did not produce {src}')
    _validate_polynomial_files([source / name for name in names])
    polynomial_cache_metadata_path().unlink(missing_ok=True)
    for name in names:
        src = source / name
        fd, tmp_name = tempfile.mkstemp(prefix=f'.{name}.', dir=str(destination))
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, destination / name)
        finally:
            tmp.unlink(missing_ok=True)
    from nimble.opensimad.mint_settings import MINT_POLYNOMIAL_BOUNDS
    payload = {
        'artifact_type': 'opensimad_polynomial_cache',
        'pipeline_version': POLYNOMIAL_PIPELINE_VERSION,
        'model': OPENSIM_MODEL_BASENAME,
        'model_sha256': _sha256(ad_scaled_adjusted_model_path()),
        'artifacts': {name: _sha256(destination / name) for name in names},
        'polynomial_bounds_degrees': MINT_POLYNOMIAL_BOUNDS,
        'complete': True,
    }
    path = polynomial_cache_metadata_path()
    _write_json_atomic(path, payload)
    return path

def validate_opensimad_worker_artifacts(*, load_library: bool=True, deep: bool=True) -> dict:
    return {
        'external_function': validate_compiled_external(load_library=load_library, require_metadata=True, deep=deep),
        'polynomial_cache': validate_polynomial_cache(require_metadata=True, deep=deep),
    }

def stage_opensimad_polynomial_cache(model_folder: Path) -> int:
    """Copy shared polynomial / MT-parameter caches into a per-segment Model dir.

    Without this, every segment re-runs MuscleAnalysis (joblib×host CPUs) and OOMs.
    """
    import shutil
    src = opensimad_dir()
    model_folder = Path(model_folder)
    model_folder.mkdir(parents=True, exist_ok=True)
    copied = 0
    validate_polynomial_cache(require_metadata=True, deep=False)
    for name in runtime_polynomial_cache_names():
        path = src / name
        dest = model_folder / name
        shutil.copy2(path, dest)
        copied += 1
    return copied

def vendor_opencap_ad_dir() -> Path:
    return Path(__file__).resolve().parent / 'vendor' / 'opencap_ad'

def vendor_dummy_motion_path() -> Path:
    return Path(__file__).resolve().parent / 'vendor' / 'opencap_pipeline' / 'MuscleAnalysis' / 'DummyMotion.mot'
