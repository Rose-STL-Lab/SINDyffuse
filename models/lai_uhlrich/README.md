# LaiUhlrich2022

Generic OpenCap full-body model (Rajagopal lineage; welded wrists).

- `LaiUhlrich2022.osim` — unscaled generic model (committed)
- `Geometry/` — optional mesh folder created by `nimble.opensimad.paths.ensure_lai_geometry()` (not committed). Prefer a local Geometry tree next to the model, or set `LAI_GEOMETRY_SRC` to an existing OpenSim/OpenCap Geometry directory. Meshes are not required for IK / OpenSimAD solves.
- `opensimad/` — generated AD base/contacts models and ExternalFunction artifacts (gitignored except `.gitignore`)

Build AD artifacts once:

```bash
python scripts/build_lai_opensimad_ext.py
```
