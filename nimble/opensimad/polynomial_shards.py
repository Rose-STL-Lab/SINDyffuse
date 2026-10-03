"""Immutable polynomial inputs and persistent, independently retryable slices."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
import shutil
import sys
import tempfile
from zipfile import BadZipFile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from common.memory_diagnostics import diagnostic_event, diagnostic_stage
from nimble.opensimad import OPENSIM_MODEL_BASENAME
from nimble.opensimad.mint_settings import MINT_POLYNOMIAL_BOUNDS
from nimble.opensimad.paths import vendor_opencap_ad_dir

MODEL_NAME = f'{OPENSIM_MODEL_BASENAME}_scaled_adjusted'
# Known predecessor: fitting-only 5D basis bugfix, extraction/data format unchanged.
_SIX_DIMENSION_PREDECESSOR = {
    'polynomial_shards.py': '798fee30658d7e48a40c6492e78790d9f7545672666cff02f31f6cae1ca4cca4',
    'polynomialsOpenSimAD.py': 'bc1cc5fcc5bdc7d8a22b08d2bdcd2dadb5081775aad4d78ff627534b0987f7e4',
}

def validate_code_identity(recorded: dict, *, allow_fitting_upgrade: bool=False) -> None:
    current = code_identity()
    changed = {key for key in recorded.keys() | current.keys() if recorded.get(key) != current.get(key)}
    if not changed:
        return
    if (allow_fitting_upgrade and changed <= _SIX_DIMENSION_PREDECESSOR.keys()
            and all(recorded.get(key) == value for key, value in _SIX_DIMENSION_PREDECESSOR.items())):
        diagnostic_event('fitting_code_upgrade', original_code=recorded, finalization_code=current,
                         reason='Extend total-degree basis/evaluator beyond five coordinates; extraction unchanged')
        return
    raise ValueError('Code/runtime changed since preparation; select a new build directory')

def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def build_identity(manifest: dict) -> str:
    payload = {key: value for key, value in manifest.items() if key != 'build_id'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def runtime_identity() -> dict:
    import casadi
    import opensim
    return {'python': sys.version, 'numpy': np.__version__, 'casadi': casadi.__version__,
            'opensim': str(getattr(opensim, '__version__', 'unknown')),
            'compute_threads': {key: os.environ.get(key, '') for key in
                                ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}}

def code_identity() -> dict:
    base = Path(__file__).resolve().parent
    paths = [Path(__file__), base / 'polynomial_cache.py', base / 'mint_settings.py', base / 'model_prep.py',
             vendor_opencap_ad_dir() / 'muscleDataOpenSimAD.py',
             vendor_opencap_ad_dir() / 'polynomialsOpenSimAD.py',
             vendor_opencap_ad_dir() / 'utils.py']
    return {path.name: file_hash(path) for path in paths}

def chunk_range(manifest: dict, index: int) -> tuple[int, int]:
    if not 0 <= index < manifest['chunk_count']:
        raise ValueError(f'Chunk index {index} outside [0, {manifest["chunk_count"]})')
    start = index * manifest['chunk_frames']
    return start, min(start + manifest['chunk_frames'], manifest['sample_count'])

@contextmanager
def build_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)

def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as fp:
            np.savez(fp, **arrays)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)

def prepare_shards(build_dir: Path, *, chunk_frames: int, expected_chunks: int | None=None) -> dict:
    """Publish immutable inputs, or reuse identical inputs without deleting results."""
    from nimble.opensimad.polynomial_cache import _ensure_vendor_on_path, _write_full_range_dummy
    from nimble.opensimad.model_prep import ensure_ad_ready_artifacts
    from nimble.opensimad.paths import ad_scaled_adjusted_model_path, validate_compiled_external, vendor_dummy_motion_path
    import opensim
    if chunk_frames <= 0:
        raise ValueError('chunk_frames must be positive')
    root = Path(build_dir).expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with build_lock(root.parent / f'.{root.name}.prepare.lock'):
        ensure_ad_ready_artifacts(force=False)
        validate_compiled_external(load_library=True, require_metadata=True, deep=True)
        _ensure_vendor_on_path()
        with tempfile.TemporaryDirectory(prefix=f'.{root.name}.', dir=root.parent) as temporary:
            stage = Path(temporary)
            model_path = stage / f'{MODEL_NAME}.osim'
            shutil.copy2(ad_scaled_adjusted_model_path(), model_path)
            motion = _write_full_range_dummy(vendor_dummy_motion_path(), stage / 'DummyMotionFullRange.mot')
            table = opensim.TimeSeriesTable(str(motion))
            if table.getTableMetaDataString('inDegrees') != 'yes':
                raise ValueError('Prepared polynomial samples must declare degrees')
            labels = [str(label) for label in table.getColumnLabels()]
            values = np.asarray(table.getMatrix().to_numpy(), dtype=np.float64)
            if not len(values) or not np.isfinite(values).all():
                raise ValueError('Fitting samples must be nonempty and finite')
            model = opensim.Model(str(model_path))
            forces = model.getForceSet()
            muscles = [forces.get(i).getName() for i in range(forces.getSize())
                       if 'Muscle' in forces.get(i).getConcreteClassName()]
            if any('Muscle' in forces.get(i).getConcreteClassName() and
                   forces.get(i).getConcreteClassName() != 'Millard2012EquilibriumMuscle'
                   for i in range(forces.getSize())):
                raise ValueError('Unexpected muscle type: cannot preserve fitting muscle order')
            atomic_npz(stage / 'samples.npz', coordinate_values=values)
            manifest = {
                'schema_version': 1, 'model_name': MODEL_NAME, 'rng_seed': 42,
                'coordinate_units': 'degrees',
                'bounds_degrees': MINT_POLYNOMIAL_BOUNDS, 'chunk_frames': chunk_frames,
                'sample_count': len(values), 'chunk_count': (len(values) + chunk_frames - 1) // chunk_frames,
                'coordinate_labels': labels, 'muscle_names': muscles,
                'model_sha256': file_hash(model_path), 'motion_sha256': file_hash(motion),
                'samples_sha256': file_hash(stage / 'samples.npz'),
                'source_motion_sha256': file_hash(vendor_dummy_motion_path()),
                'runtime': runtime_identity(), 'code': code_identity(),
                'fitting': {'order_min': 3, 'order_max': 9, 'threshold': 0.0015,
                            'removeBadHipFlexionEntries': True, 'dtype': 'float64'},
            }
            if expected_chunks is not None and manifest['chunk_count'] != expected_chunks:
                raise ValueError(f'Expected {expected_chunks} tasks, but samples require {manifest["chunk_count"]}')
            manifest['build_id'] = build_identity(manifest)
            if (root / 'manifest.json').exists():
                old, _ = load_inputs(root)
                if old != manifest:
                    raise ValueError('Build inputs changed. Select a new --build_dir; existing results are immutable.')
                diagnostic_event('prepared_inputs_reused', build_id=manifest['build_id'])
                return manifest
            if root.exists():
                raise ValueError(f'Unpublished build directory already exists: {root}; select a new --build_dir')
            (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            # Same-filesystem rename publishes the complete input snapshot at once.
            os.rename(stage, root)
    diagnostic_event('prepared_inputs', build_id=manifest['build_id'], chunk_count=manifest['chunk_count'])
    return manifest

def load_inputs(build_dir: Path, *, expected_chunks: int | None=None, allow_fitting_upgrade: bool=False) -> tuple[dict, np.ndarray]:
    root = Path(build_dir)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('build_id') != build_identity(manifest):
        raise ValueError('Invalid build manifest identity/version')
    if manifest['chunk_frames'] <= 0 or manifest['sample_count'] <= 0:
        raise ValueError('Invalid sample/chunk count')
    if manifest['chunk_count'] != (manifest['sample_count'] + manifest['chunk_frames'] - 1) // manifest['chunk_frames']:
        raise ValueError('Manifest chunk count does not cover samples')
    if expected_chunks is not None and manifest['chunk_count'] != expected_chunks:
        raise ValueError('Indexed Job task count does not match prepared inputs')
    for name, key in ((f'{MODEL_NAME}.osim', 'model_sha256'), ('DummyMotionFullRange.mot', 'motion_sha256'), ('samples.npz', 'samples_sha256')):
        if file_hash(root / name) != manifest[key]:
            raise ValueError(f'Input checksum mismatch: {name}')
    validate_code_identity(manifest['code'], allow_fitting_upgrade=allow_fitting_upgrade)
    if manifest['runtime'] != runtime_identity():
        raise ValueError('Code/runtime changed since preparation; select a new build directory')
    with np.load(root / 'samples.npz', allow_pickle=False) as archive:
        values = archive['coordinate_values'].copy()
    if values.shape != (manifest['sample_count'], len(manifest['coordinate_labels'])) or not np.isfinite(values).all():
        raise ValueError('Invalid coordinate samples')
    return manifest, values

def chunk_path(root: Path, index: int) -> Path:
    return root / 'chunks' / f'chunk_{index:06d}.npz'

def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def validate_chunk(path: Path, manifest: dict, values: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    start, end = chunk_range(manifest, index)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'].item()))
        expected = {'build_id': manifest['build_id'], 'index': index, 'start': start, 'end': end,
                    'lengths_sha256': array_hash(archive['mtu_lengths']),
                    'moment_arms_sha256': array_hash(archive['mtu_moment_arms'])}
        if metadata != expected:
            raise ValueError(f'Stale or misindexed result: {path}')
        coordinates = archive['coordinate_values']
        lmt, moment_arms = archive['mtu_lengths'].copy(), archive['mtu_moment_arms'].copy()
    n, m, c = end - start, len(manifest['muscle_names']), len(manifest['coordinate_labels'])
    if lmt.shape != (n, m) or moment_arms.shape != (n, m, c):
        raise ValueError(f'Invalid result shapes: {path}')
    if not np.array_equal(coordinates, values[start:end]) or not np.isfinite(lmt).all() or not np.isfinite(moment_arms).all():
        raise ValueError(f'Invalid result values: {path}')
    if lmt.dtype != np.float64 or moment_arms.dtype != np.float64:
        raise ValueError(f'Invalid result precision: {path}')
    return lmt, moment_arms

def extract_shard(build_dir: Path, *, index: int, work_dir: Path, expected_chunks: int | None=None) -> Path:
    from nimble.opensimad.polynomial_cache import _ensure_vendor_on_path
    _ensure_vendor_on_path()
    from muscleDataOpenSimAD import _get_mtu_chunk_isolated
    root = Path(build_dir).resolve()
    manifest, values = load_inputs(root, expected_chunks=expected_chunks)
    start, end = chunk_range(manifest, index)
    result = chunk_path(root, index)
    with build_lock(root / 'chunks' / f'.chunk_{index:06d}.lock'):
        if result.exists():
            try:
                validate_chunk(result, manifest, values, index)
            except (ValueError, OSError, KeyError, EOFError, BadZipFile) as exc:
                diagnostic_event('invalid_chunk_recomputed', index=index, error=repr(exc))
            else:
                diagnostic_event('chunk_reused', index=index, build_id=manifest['build_id'])
                return result
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f'chunk_{index:06d}_', dir=work_dir) as scratch:
            model = Path(scratch) / f'{MODEL_NAME}.osim'
            shutil.copy2(root / model.name, model)
            with diagnostic_stage('distributed_chunk', chunk_index=index, start_frame=start, end_frame=end,
                                  build_id=manifest['build_id']):
                lmt, moment_arms = _get_mtu_chunk_isolated((str(model), values[start:end], manifest['coordinate_labels'], index))
                lmt, moment_arms = np.asarray(lmt, dtype=np.float64), np.asarray(moment_arms, dtype=np.float64)
                metadata = {'build_id': manifest['build_id'], 'index': index, 'start': start, 'end': end,
                            'lengths_sha256': array_hash(lmt), 'moment_arms_sha256': array_hash(moment_arms)}
                # Validate a private candidate before making it visible to finalization.
                fd, candidate_name = tempfile.mkstemp(prefix=f'.chunk_{index:06d}.', dir=result.parent)
                os.close(fd)
                candidate = Path(candidate_name)
                try:
                    atomic_npz(candidate, metadata=np.array(json.dumps(metadata)), coordinate_values=values[start:end],
                               mtu_lengths=lmt, mtu_moment_arms=moment_arms)
                    validate_chunk(candidate, manifest, values, index)
                    os.replace(candidate, result)
                finally:
                    candidate.unlink(missing_ok=True)
    return result

def assemble_shards(build_dir: Path, *, expected_chunks: int | None=None, allow_fitting_upgrade: bool=False) -> tuple[dict, dict]:
    root = Path(build_dir)
    manifest, values = load_inputs(root, expected_chunks=expected_chunks, allow_fitting_upgrade=allow_fitting_upgrade)
    lengths, arms = [], []
    for index in range(manifest['chunk_count']):
        lmt, moment_arms = validate_chunk(chunk_path(root, index), manifest, values, index)
        lengths.append(lmt)
        arms.append(moment_arms)
    return manifest, {'mtu_lengths': np.concatenate(lengths, axis=0),
                      'mtu_moment_arms': np.concatenate(arms, axis=0),
                      'muscle_names': manifest['muscle_names'],
                      'coordinate_names': [label.split('/')[-2] for label in manifest['coordinate_labels']],
                      'coordinate_values': values}

def finalize_shards(build_dir: Path, *, work_dir: Path, expected_chunks: int | None=None, allow_fitting_upgrade: bool=False) -> dict:
    from nimble.opensimad.polynomial_cache import _ensure_vendor_on_path, RIGHT_JOINTS, LEFT_JOINTS, RIGHT_MUSCLES, LEFT_MUSCLES
    from nimble.opensimad.paths import ad_scaled_adjusted_model_path, opensimad_dir, publish_polynomial_cache, validate_polynomial_cache
    _ensure_vendor_on_path()
    from muscleDataOpenSimAD import getMTParameters, getPolynomialData
    root = Path(build_dir).resolve()
    with build_lock(root / '.finalize.lock'), diagnostic_stage('finalize_shards'):
        manifest, fitting = assemble_shards(root, expected_chunks=expected_chunks, allow_fitting_upgrade=allow_fitting_upgrade)
        if file_hash(ad_scaled_adjusted_model_path()) != manifest['model_sha256']:
            raise ValueError('Shared runtime model changed since preparation; refusing to publish')
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='polynomial_finalize_', dir=work_dir) as scratch:
            model_dir = Path(scratch)
            model = model_dir / f'{MODEL_NAME}.osim'
            shutil.copy2(root / model.name, model)
            np.save(model_dir / f'data4PolynomialFitting_{MODEL_NAME}_default.npy', fitting)
            for side, joints, muscles in (('r', RIGHT_JOINTS, RIGHT_MUSCLES), ('l', LEFT_JOINTS, LEFT_MUSCLES)):
                getMTParameters(str(model), list(muscles), False, pathMTParameters=str(model_dir), modelName=MODEL_NAME, side=side)
                getPolynomialData(False, str(model_dir), modelName=MODEL_NAME, joints=list(joints), muscles=list(muscles), side=side)
            # Serialize final publication even when different build directories finalize.
            with build_lock(opensimad_dir() / '.polynomial-publish.lock'):
                if file_hash(ad_scaled_adjusted_model_path()) != manifest['model_sha256']:
                    raise ValueError('Shared runtime model changed during fitting; refusing to publish')
                metadata_path = publish_polynomial_cache(model_dir)
                metadata = json.loads(metadata_path.read_text())
                metadata['distributed_build_id'] = manifest['build_id']
                metadata['distributed_manifest_sha256'] = file_hash(root / 'manifest.json')
                metadata['finalization_code'] = code_identity()
                metadata['fitting_code_upgrade_allowed'] = allow_fitting_upgrade
                fd, name = tempfile.mkstemp(prefix='.polynomial-metadata.', dir=metadata_path.parent)
                try:
                    with os.fdopen(fd, 'w') as fp:
                        json.dump(metadata, fp, indent=2)
                        fp.write('\n')
                        fp.flush()
                        os.fsync(fp.fileno())
                    os.replace(name, metadata_path)
                finally:
                    Path(name).unlink(missing_ok=True)
                metadata = validate_polynomial_cache(require_metadata=True, deep=True)
        result = {'metadata_path': str(metadata_path), 'build_id': manifest['build_id'],
                  'finalization_code': code_identity(), 'artifacts': sorted(metadata['artifacts'])}
        (root / 'finalized.json').write_text(json.dumps(result, indent=2) + '\n')
        return result