"""SSM67 / MinT virtual marker definitions (SMPL-H vertex indices + Lai templates)."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

from mint.coord_map import default_mint_model_path

MARKER_NAMES: Tuple[str, ...] = (
    "C7",
    "CLAV",
    "LANK",
    "LASI",
    "LBAK",
    "LBCEP",
    "LBHD",
    "LBUM",
    "LBUST",
    "LCHEECK",
    "LELB",
    "LELBIN",
    "LFHD",
    "LFIN",
    "LFRM",
    "LFTHIIN",
    "LHEE",
    "LIWR",
    "LKNE",
    "LKNI",
    "LMT1",
    "LMT5",
    "LNWST",
    "LOWR",
    "LPSI",
    "LRSTBEEF",
    "LSCAP",
    "LSHN",
    "LSHO",
    "LTHI",
    "LTHILO",
    "LTHMB",
    "LTOE",
    "MBLLY",
    "RANK",
    "RASI",
    "RBAK",
    "RBCEP",
    "RBHD",
    "RBSH",
    "RBUM",
    "RBUST",
    "RCHEECK",
    "RELB",
    "RELBIN",
    "RFHD",
    "RFIN",
    "RFRM2",
    "RFTHI",
    "RFTHIIN",
    "RHEE",
    "RIWR",
    "RKNE",
    "RKNI",
    "RMT1",
    "RMT5",
    "RNWST",
    "ROWR",
    "RPSI",
    "RRSTBEEF",
    "RSHO",
    "RTHI",
    "RTHMB",
    "RTIB",
    "RTOE",
    "STRN",
    "T8",
)

VERTEX_INDICES: Tuple[int, ...] = (
    3470,
    3171,
    3327,
    857,
    1812,
    628,
    182,
    3116,
    3040,
    239,
    1666,
    1725,
    0,
    2174,
    1568,
    1368,
    3387,
    2112,
    1053,
    1058,
    3336,
    3346,
    1323,
    2108,
    3122,
    3314,
    1252,
    1082,
    1861,
    1454,
    850,
    2224,
    3233,
    1769,
    6728,
    4343,
    5273,
    4116,
    3694,
    6399,
    6540,
    6488,
    3749,
    5135,
    5194,
    3512,
    5635,
    5210,
    4360,
    4841,
    6786,
    5573,
    4538,
    4544,
    6736,
    6747,
    4804,
    5568,
    6544,
    6682,
    5322,
    4927,
    5686,
    4598,
    6633,
    3506,
    3508,
)

SSM67_TO_OPENCAP_TEMPLATE: Dict[str, str] = {
    "C7": "C7",
    "CLAV": "R_Sternum",
    "LANK": "L_ankle",
    "LASI": "L.ASIS",
    "LBAK": "L_Sternum",
    "LBCEP": "L_elbow_lat",
    "LBHD": "C7",
    "LBUM": "L.ASIS",
    "LBUST": "L_Sternum",
    "LCHEECK": "C7",
    "LELB": "L_elbow_lat",
    "LELBIN": "L_elbow_med",
    "LFHD": "C7",
    "LFIN": "L_wrist_radius",
    "LFRM": "L_wrist_radius",
    "LFTHIIN": "L_thigh1",
    "LHEE": "L_calc",
    "LIWR": "L_wrist_radius",
    "LKNE": "L_knee",
    "LKNI": "L_mknee",
    "LMT1": "L_toe",
    "LMT5": "L_5meta",
    "LNWST": "L_Sternum",
    "LOWR": "L_wrist_ulna",
    "LPSI": "L.PSIS",
    "LRSTBEEF": "L_calc",
    "LSCAP": "L_Shoulder",
    "LSHN": "L_shank_antsup",
    "LSHO": "L_Shoulder",
    "LTHI": "L_thigh2",
    "LTHILO": "L_thigh3",
    "LTHMB": "L_wrist_radius",
    "LTOE": "L_toe",
    "MBLLY": "R_Sternum",
    "RANK": "r_ankle",
    "RASI": "r.ASIS",
    "RBAK": "R_Sternum",
    "RBCEP": "R_elbow_lat",
    "RBHD": "C7",
    "RBSH": "R_Sternum",
    "RBUM": "r.ASIS",
    "RBUST": "R_Sternum",
    "RCHEECK": "C7",
    "RELB": "R_elbow_lat",
    "RELBIN": "R_elbow_med",
    "RFHD": "C7",
    "RFIN": "R_wrist_radius",
    "RFRM2": "R_wrist_radius",
    "RFTHI": "r_thigh1",
    "RFTHIIN": "r_thigh2",
    "RHEE": "r_calc",
    "RIWR": "R_wrist_radius",
    "RKNE": "r_knee",
    "RKNI": "r_mknee",
    "RMT1": "r_toe",
    "RMT5": "r_5meta",
    "RNWST": "R_Sternum",
    "ROWR": "R_wrist_ulna",
    "RPSI": "r.PSIS",
    "RRSTBEEF": "r_calc",
    "RSHO": "R_Shoulder",
    "RTHI": "r_thigh2",
    "RTHMB": "R_wrist_radius",
    "RTIB": "r_shank_antsup",
    "RTOE": "r_toe",
    "STRN": "R_Sternum",
    "T8": "C7",
}

PELVIS_MARKERS = frozenset({"RASI", "LASI", "RPSI", "LPSI", "RBUM", "LBUM"})


def marker_vertex_map() -> Dict[str, int]:
    return dict(zip(MARKER_NAMES, VERTEX_INDICES))


def opensim_assets_dir() -> Path:
    return default_mint_model_path().parent / "opensim"


def _parse_vec3(text: str) -> Tuple[float, float, float]:
    parts = [float(x) for x in str(text).split()]
    if len(parts) != 3:
        raise ValueError(f"Expected vec3, got {text!r}")
    return parts[0], parts[1], parts[2]


def _offset_for_name(name: str) -> Tuple[float, float, float]:
    """Small unique local offset so shared templates remain distinguishable in IK."""
    h = hashlib.md5(name.encode("utf-8")).digest()
    scale = 0.002  # 2 mm
    return (
        (h[0] / 255.0 - 0.5) * scale,
        (h[1] / 255.0 - 0.5) * scale,
        (h[2] / 255.0 - 0.5) * scale,
    )


@lru_cache(maxsize=1)
def _load_opencap_template_markers() -> Dict[str, Tuple[str, Tuple[float, float, float]]]:
    path = opensim_assets_dir() / "LaiUhlrich2022_markers_mocap.xml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing OpenCap marker template: {path}")
    root = ET.parse(path).getroot()
    out: Dict[str, Tuple[str, Tuple[float, float, float]]] = {}
    for marker in root.iter("Marker"):
        name = marker.get("name")
        if not name:
            continue
        frame = marker.find("socket_parent_frame")
        loc = marker.find("location")
        if frame is None or loc is None:
            continue
        out[str(name)] = (str(frame.text).strip(), _parse_vec3(loc.text or "0 0 0"))
    return out


def ensure_ssm67_marker_xml() -> Path:
    """Build Lai marker set with 67 SSM names (cached on disk)."""
    out_path = opensim_assets_dir() / "LaiUhlrich2022_markers_ssm67.xml"
    if out_path.is_file():
        return out_path

    templates = _load_opencap_template_markers()
    lines = [
        '<?xml version="1.0" encoding="UTF-8" ?>',
        '<OpenSimDocument Version="40000">',
        '\t<MarkerSet name="markerset">',
        "\t\t<objects>",
    ]
    for name in MARKER_NAMES:
        template_name = SSM67_TO_OPENCAP_TEMPLATE.get(name, "C7")
        if template_name not in templates:
            raise KeyError(f"Template marker {template_name!r} missing for {name!r}")
        frame, loc = templates[template_name]
        ox, oy, oz = _offset_for_name(name)
        lx, ly, lz = loc[0] + ox, loc[1] + oy, loc[2] + oz
        lines.extend(
            [
                f'\t\t\t<Marker name="{name}">',
                f"\t\t\t\t<socket_parent_frame>{frame}</socket_parent_frame>",
                f"\t\t\t\t<location>{lx} {ly} {lz}</location>",
                "\t\t\t\t<fixed>false</fixed>",
                "\t\t\t</Marker>",
            ]
        )
    lines.extend(["\t\t</objects>", "\t\t<groups />", "\t</MarkerSet>", "</OpenSimDocument>"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def ensure_ssm67_ik_setup_xml() -> Path:
    out_path = opensim_assets_dir() / "Setup_IK_ssm67.xml"
    if out_path.is_file():
        return out_path
    tasks = []
    for name in MARKER_NAMES:
        w = 25 if name in PELVIS_MARKERS else 5
        tasks.append(
            "\n".join(
                [
                    f'\t\t\t\t<IKMarkerTask name="{name}">',
                    "\t\t\t\t\t<apply>true</apply>",
                    f"\t\t\t\t\t<weight>{w}</weight>",
                    "\t\t\t\t</IKMarkerTask>",
                ]
            )
        )
    body = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8" ?>',
            '<OpenSimDocument Version="40000">',
            '\t<InverseKinematicsTool name="ssm67">',
            "\t\t<results_directory>Unassigned</results_directory>",
            "\t\t<model_file>Unassigned</model_file>",
            "\t\t<constraint_weight>Inf</constraint_weight>",
            "\t\t<accuracy>1e-5</accuracy>",
            "\t\t<IKTaskSet>",
            "\t\t\t<objects>",
            ",\n".join(tasks),
            "\t\t\t</objects>",
            "\t\t\t<groups />",
            "\t\t</IKTaskSet>",
            "\t\t<marker_file>Unassigned</marker_file>",
            "\t\t<coordinate_file>Unassigned</coordinate_file>",
            "\t\t<report_errors>true</report_errors>",
            "\t\t<report_marker_locations>false</report_marker_locations>",
            "\t\t<output_motion_file>Unassigned</output_motion_file>",
            "\t\t<output_marker_file>Unassigned</output_marker_file>",
            "\t\t<output_forces_file>Unassigned</output_forces_file>",
            "\t</InverseKinematicsTool>",
            "</OpenSimDocument>",
        ]
    )
    out_path.write_text(body, encoding="utf-8")
    return out_path
