"""Soft-import opensim.moco so Python casadi can load in the same process.

Moco still works in a fresh interpreter that does not import casadi first
(OpenSim's bundled libcasadi.so.3.7 ABI).
"""
from pathlib import Path

p = Path("/opt/conda/envs/sindyffuse/lib/python3.11/site-packages/opensim/__init__.py")
t = p.read_text()
old = "from .moco import *"
new = """try:
    from .moco import *
except ImportError as _e:
    import warnings
    warnings.warn("opensim.moco unavailable: %s" % (_e,), stacklevel=1)"""
if "opensim.moco unavailable" in t:
    print("already patched")
elif old not in t:
    raise SystemExit("opensim moco import not found in %s" % (p,))
else:
    p.write_text(t.replace(old, new, 1))
    print("patched opensim.moco soft-import")
