# MinT OpenSim models

## Lai lower-body model (bundled)

`LaiUhlrich2022.osim` — OpenCap / Lai et al. musculoskeletal model used by MinT for lower-body simulation (80 LU muscles).

Source: [opencap-core](https://github.com/stanfordnmbl/opencap-core/blob/main/opensimPipeline/Models/LaiUhlrich2022.osim)

## Bruno thoracolumbar model (manual download)

MinT combines the Lai lower-body model with the Bruno et al. thoracolumbar spine model (322 TL muscle strands). Place the Bruno `.osim` file here as:

```
osim/models/BrunoThoracolumbar.osim
```

Obtain from the [MinT dataset release](https://radar.kit.edu/radar/en/dataset/VDPCEFSThBWlDPFL) or paper supplement.

Until the combined model is available, preprocess uses the Lai model for IK (`q` ndof ≈ 33) while muscle labels still use the full 402 MinT activations from the dataset.

## SSM67 virtual markers (MinT retargeting)

Retargeting samples **67 SSM markers** from SMPL-H mesh vertices (Bittner/MinT convention), then runs OpenSim IK on the Lai model:

- `ssm67_smplh_markers.yaml` — marker names + SMPL-H vertex indices
- `opensim/LaiUhlrich2022_markers_mocap.xml` — OpenCap 49-marker template (source)
- `opensim/LaiUhlrich2022_markers_ssm67.xml` — generated 67-marker Lai attachment set
- `opensim/Setup_IK_ssm67.xml` — IK task weights (pelvis markers weighted higher)

Set `SMPLH_MODEL_PATH` to a directory containing `SMPLH_NEUTRAL.pkl`. Scaling is skipped in v1 (standard unscaled Lai model, matching MinT paper).

## Override path

```bash
export MINT_OSIM_PATH=/path/to/custom.osim
```

## OpenSim version

SINDyffuse uses OpenSim 4.5.2 (conda). MinT was generated with OpenSim 4.0; test model compatibility after upgrades.
