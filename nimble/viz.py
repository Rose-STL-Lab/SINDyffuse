"""Rajagopal mesh visualization helpers for B3D motion figures."""

from __future__ import annotations

import struct
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

MeshSpec = Tuple[str, str]
ViewPreset = Tuple[str, float, float, float]  # (name, elev, azim, roll)

DEFAULT_MESH_COLOR = (0.75, 0.75, 0.78, 0.95)
DEFAULT_GROUND_COLOR = (0.5, 0.5, 0.5, 0.5)
DEFAULT_AXIS_LIMIT = 1.2

# Montage poses are spaced along world +X; ground is the XZ plane at y=0.
VIEW_PRESETS: Dict[str, ViewPreset] = {
    "default": ("default", 110.0, -90.0, 0.0),
    # Level view along ±Z so montage spread (+X) reads left-to-right on screen.
    "flat": ("flat", 90, -90.0, 0.0),
}


def resolve_view_preset(name: str) -> ViewPreset:
    key = str(name).strip().lower()
    if key not in VIEW_PRESETS:
        choices = ", ".join(sorted(VIEW_PRESETS))
        raise ValueError(f"Unknown view {name!r}; choose from: {choices}")
    return VIEW_PRESETS[key]


def view_angles(view: str | ViewPreset | None = None) -> Tuple[float, float, float]:
    if view is None:
        preset = VIEW_PRESETS["default"]
    elif isinstance(view, str):
        preset = resolve_view_preset(view)
    else:
        preset = view
    roll = float(preset[3]) if len(preset) > 3 else 0.0
    return float(preset[1]), float(preset[2]), roll


def apply_view(ax: Any, view: str | ViewPreset | None = None) -> None:
    elev, azim, roll = view_angles(view)
    try:
        ax.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        ax.view_init(elev=elev, azim=azim)


def _triangulate_face(indices: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Fan-triangulate a polygon face into triangles."""
    if len(indices) < 3:
        return []
    if len(indices) == 3:
        return [(int(indices[0]), int(indices[1]), int(indices[2]))]
    v0 = int(indices[0])
    return [
        (v0, int(indices[i]), int(indices[i + 1]))
        for i in range(1, len(indices) - 1)
    ]


def read_ply_mesh(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a binary little-endian PLY and triangulate polygon faces."""
    ply_path = Path(path)
    with ply_path.open("rb") as f:
        n_verts = 0
        n_faces = 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
            elif line == "end_header":
                break

        verts = np.frombuffer(f.read(n_verts * 12), dtype="<f4").reshape(n_verts, 3)
        faces: List[Tuple[int, int, int]] = []
        for _ in range(n_faces):
            n_indices = f.read(1)[0]
            indices = struct.unpack("<" + "i" * n_indices, f.read(4 * n_indices))
            faces.extend(_triangulate_face(indices))

    if not faces:
        return verts.astype(np.float64), np.zeros((0, 3), dtype=np.int32)
    return verts.astype(np.float64), np.asarray(faces, dtype=np.int32)


@lru_cache(maxsize=256)
def _cached_ply(path: str) -> Tuple[np.ndarray, np.ndarray]:
    return read_ply_mesh(path)


def rajagopal_mesh_specs(skeleton: Any) -> List[MeshSpec]:
    """Return ``(body_name, ply_path)`` for every mesh shape on the skeleton."""
    specs: List[MeshSpec] = []
    for body_idx in range(int(skeleton.getNumBodyNodes())):
        body = skeleton.getBodyNode(body_idx)
        body_name = str(body.getName())
        for shape_idx in range(int(body.getNumShapeNodes())):
            shape = body.getShapeNode(shape_idx).getShape()
            if not hasattr(shape, "asMeshShape"):
                continue
            mesh_shape = shape.asMeshShape()
            if mesh_shape is None:
                continue
            mesh_path = str(mesh_shape.getMeshPath()).strip()
            if mesh_path:
                specs.append((body_name, mesh_path))
    return specs


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape == (4, 4):
        hom = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
        return (hom @ mat.T)[:, :3]
    raise ValueError(f"Expected 4x4 transform matrix, got {mat.shape}")


def _pelvis_world_xy(skeleton: Any) -> Tuple[float, float]:
    body = skeleton.getBodyNode("pelvis")
    transform = body.getWorldTransform()
    if hasattr(transform, "translation"):
        origin = np.asarray(transform.translation(), dtype=np.float64).reshape(3)
    else:
        origin = np.asarray(transform.matrix(), dtype=np.float64)[:3, 3]
    return float(origin[0]), float(origin[2])


def rajagopal_world_meshes(
    skeleton: Any,
    q_frame: np.ndarray,
    mesh_specs: Sequence[MeshSpec],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Pose the skeleton and return world-space mesh vertices and triangle faces."""
    skeleton.setPositions(np.asarray(q_frame, dtype=np.float64).reshape(-1))
    skeleton.computeForwardKinematics()

    vertices_list: List[np.ndarray] = []
    faces_list: List[np.ndarray] = []
    for body_name, ply_path in mesh_specs:
        local_vertices, faces = _cached_ply(ply_path)
        if faces.size == 0:
            continue
        body = skeleton.getBodyNode(str(body_name))
        world_vertices = _transform_points(
            np.asarray(body.getWorldTransform().matrix(), dtype=np.float64),
            local_vertices,
        )
        vertices_list.append(world_vertices)
        faces_list.append(faces)
    return vertices_list, faces_list


def normalize_pose_meshes(
    vertices_list: Sequence[np.ndarray],
    *,
    height_offset: float,
    root_x: float,
    root_z: float,
) -> List[np.ndarray]:
    """Floor-align and pelvis-center mesh vertices for display."""
    out: List[np.ndarray] = []
    for vertices in vertices_list:
        posed = np.asarray(vertices, dtype=np.float64).copy()
        posed[:, 1] -= float(height_offset)
        posed[:, 0] -= float(root_x)
        posed[:, 2] -= float(root_z)
        out.append(posed)
    return out


def mesh_bounds(vertices_list: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not vertices_list:
        zeros = np.zeros(3, dtype=np.float64)
        return zeros, zeros
    stacked = np.concatenate([np.asarray(v, dtype=np.float64) for v in vertices_list], axis=0)
    return stacked.min(axis=0), stacked.max(axis=0)


def add_ground_plane(
    ax: Any,
    *,
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
    y: float = 0.0,
    facecolor: Tuple[float, float, float, float] = DEFAULT_GROUND_COLOR,
) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = [
        [min_x, y, min_z],
        [min_x, y, max_z],
        [max_x, y, max_z],
        [max_x, y, min_z],
    ]
    plane = Poly3DCollection([verts])
    plane.set_facecolor(facecolor)
    ax.add_collection3d(plane)


def draw_rajagopal_pose(
    ax: Any,
    skeleton: Any,
    q_frame: np.ndarray,
    mesh_specs: Sequence[MeshSpec],
    *,
    height_offset: float,
    root_x: float,
    root_z: float,
    facecolor: Tuple[float, float, float, float] = DEFAULT_MESH_COLOR,
    ground_bounds: Tuple[float, float, float, float] | None = None,
    axis_limit: float = DEFAULT_AXIS_LIMIT,
    x_shift: float = 0.0,
    z_shift: float = 0.0,
    draw_ground: bool = True,
    configure_axes: bool = True,
    view: str | ViewPreset | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw one posed Rajagopal mesh on ``ax``; returns pose min/max bounds."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    raw_vertices, faces_list = rajagopal_world_meshes(skeleton, q_frame, mesh_specs)
    vertices_list = normalize_pose_meshes(
        raw_vertices,
        height_offset=height_offset,
        root_x=root_x,
        root_z=root_z,
    )

    for vertices, faces in zip(vertices_list, faces_list):
        posed = np.asarray(vertices, dtype=np.float64).copy()
        posed[:, 0] += float(x_shift)
        posed[:, 2] += float(z_shift)
        triangles = posed[faces]
        mesh = Poly3DCollection(triangles, linewidths=0.0)
        mesh.set_facecolor(facecolor)
        mesh.set_edgecolor((0.0, 0.0, 0.0, 0.0))
        ax.add_collection3d(mesh)

    shifted_vertices = []
    for vertices in vertices_list:
        posed = np.asarray(vertices, dtype=np.float64).copy()
        posed[:, 0] += float(x_shift)
        posed[:, 2] += float(z_shift)
        shifted_vertices.append(posed)

    mins, maxs = mesh_bounds(shifted_vertices)
    if draw_ground:
        if ground_bounds is None:
            pad = 0.15
            ground_bounds = (
                float(mins[0]) - pad,
                float(maxs[0]) + pad,
                float(mins[2]) - pad,
                float(maxs[2]) + pad,
            )
        min_x, max_x, min_z, max_z = ground_bounds
        add_ground_plane(ax, min_x=min_x, max_x=max_x, min_z=min_z, max_z=max_z)

    if configure_axes:
        limit = float(axis_limit)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(0.0, limit)
        apply_view(ax, view)
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    return mins, maxs


def _single_pose_width(
    skeleton: Any,
    q_frame: np.ndarray,
    mesh_specs: Sequence[MeshSpec],
    *,
    height_offset: float,
) -> float:
    root_x, root_z = compute_root_offsets(skeleton, q_frame)
    raw_vertices, _ = rajagopal_world_meshes(skeleton, q_frame, mesh_specs)
    vertices_list = normalize_pose_meshes(
        raw_vertices,
        height_offset=height_offset,
        root_x=root_x,
        root_z=root_z,
    )
    mins, maxs = mesh_bounds(vertices_list)
    return float(maxs[0] - mins[0])


def draw_rajagopal_montage(
    ax: Any,
    skeleton: Any,
    q_frames: Sequence[np.ndarray],
    mesh_specs: Sequence[MeshSpec],
    *,
    height_offset: float,
    pose_spacing: float | None = None,
    facecolor: Tuple[float, float, float, float] = DEFAULT_MESH_COLOR,
    view: str | ViewPreset | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw multiple poses in one 3D axes, spaced in a row on a shared ground plane.

    Returns un-padded mesh bounds ``(mins, maxs)`` for layout/cropping.
    """
    if not q_frames:
        raise ValueError("q_frames must not be empty")

    if pose_spacing is None:
        widths = [
            _single_pose_width(
                skeleton,
                q_frame,
                mesh_specs,
                height_offset=height_offset,
            )
            for q_frame in q_frames
        ]
        pose_spacing = max(widths) + 0.35

    all_mins: List[np.ndarray] = []
    all_maxs: List[np.ndarray] = []
    for pose_idx, q_frame in enumerate(q_frames):
        root_x, root_z = compute_root_offsets(skeleton, q_frame)
        mins, maxs = draw_rajagopal_pose(
            ax,
            skeleton,
            q_frame,
            mesh_specs,
            height_offset=height_offset,
            root_x=root_x,
            root_z=root_z,
            facecolor=facecolor,
            x_shift=float(pose_idx) * float(pose_spacing),
            draw_ground=False,
            configure_axes=False,
        )
        all_mins.append(mins)
        all_maxs.append(maxs)

    mins = np.min(np.stack(all_mins, axis=0), axis=0)
    maxs = np.max(np.stack(all_maxs, axis=0), axis=0)
    pad = 0.2
    add_ground_plane(
        ax,
        min_x=float(mins[0]) - pad,
        max_x=float(maxs[0]) + pad,
        min_z=float(mins[2]) - pad,
        max_z=float(maxs[2]) + pad,
    )

    # Mesh vertices are plotted as matplotlib (x, y, z) = (world_x, world_y, world_z).
    ax.set_xlim(float(mins[0]) - pad, float(maxs[0]) + pad)
    ax.set_ylim(float(mins[1]) - pad, float(maxs[1]) + pad)
    ax.set_zlim(float(mins[2]) - pad, float(maxs[2]) + pad)

    dx = float(maxs[0] - mins[0]) + 2.0 * pad
    dy = float(maxs[1] - mins[1]) + 2.0 * pad
    dz = float(maxs[2] - mins[2]) + 2.0 * pad
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((dx, dy, dz))

    apply_view(ax, view)
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    return mins, maxs


def _project_bounds_corners_2d(
    ax: Any,
    mins: Sequence[float],
    maxs: Sequence[float],
) -> np.ndarray:
    from itertools import product

    from mpl_toolkits.mplot3d import proj3d

    projection = ax.get_proj()
    lo = np.asarray(mins, dtype=np.float64)
    hi = np.asarray(maxs, dtype=np.float64)
    points_2d = []
    for x, y, z in product((lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])):
        x2d, y2d, _ = proj3d.proj_transform(x, y, z, projection)
        points_2d.append(ax.transData.transform((x2d, y2d)))
    return np.asarray(points_2d, dtype=np.float64)


def projected_bounds_aspect(ax: Any, mins: Sequence[float], maxs: Sequence[float]) -> float:
    """Return projected height / width for a 3D bounds box."""
    fig = ax.get_figure()
    fig.canvas.draw()
    pts = _project_bounds_corners_2d(ax, mins, maxs)
    width = float(pts[:, 0].max() - pts[:, 0].min())
    height = float(pts[:, 1].max() - pts[:, 1].min())
    if width <= 1e-8:
        return 1.0
    return height / width


def fit_figure_to_projected_content(
    fig: Any,
    ax: Any,
    mins: Sequence[float],
    maxs: Sequence[float],
    *,
    top_margin: float = 0.06,
    bottom_margin: float = 0.06,
) -> None:
    """Resize the figure so the axes match projected mesh bounds (less vertical gap)."""
    aspect = projected_bounds_aspect(ax, mins, maxs)
    fig_width = float(fig.get_figwidth())
    axes_height = fig_width * 0.96 * aspect
    fig_height = max(1.2, axes_height + top_margin + bottom_margin)
    fig.set_size_inches(fig_width, fig_height)

    top = 1.0 - (top_margin / fig_height)
    bottom = bottom_margin / fig_height
    ax.set_position([0.02, bottom, 0.96, top - bottom])
    fig.canvas.draw()


def add_figure_caption(
    fig: Any,
    ax: Any,
    caption: str,
    *,
    fontsize: float = 8.0,
    gap: float = 0.006,
) -> Any:
    """Place a wrapped caption just above the montage axes."""
    from textwrap import wrap

    wrap_cols = max(28, int(float(fig.get_figwidth()) * 0.96 * 7))
    wrapped = "\n".join(wrap(str(caption).strip(), width=wrap_cols))
    ax_pos = ax.get_position()
    # 3D axes do not support ax.text(..., transform=ax.transAxes); use figure coords.
    y = ax_pos.y1 + float(gap) * ax_pos.height
    text = fig.text(
        0.5,
        y,
        wrapped,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        transform=fig.transFigure,
    )
    fig.canvas.draw()
    return text


def figure_content_bbox(
    fig: Any,
    ax: Any,
    mins: Sequence[float],
    maxs: Sequence[float],
    *,
    extra_artists: Sequence[Any] = (),
    pad_pixels: float = 3.0,
) -> Any:
    """Bounding box in figure inches around projected mesh bounds and optional artists."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = _project_bounds_corners_2d(ax, mins, maxs)
    min_x = float(pts[:, 0].min()) - pad_pixels
    max_x = float(pts[:, 0].max()) + pad_pixels
    min_y = float(pts[:, 1].min()) - pad_pixels
    max_y = float(pts[:, 1].max()) + pad_pixels

    for artist in extra_artists:
        if artist is None:
            continue
        artist_bbox = artist.get_window_extent(renderer)
        min_x = min(min_x, float(artist_bbox.x0) - pad_pixels)
        max_x = max(max_x, float(artist_bbox.x1) + pad_pixels)
        min_y = min(min_y, float(artist_bbox.y0) - pad_pixels)
        max_y = max(max_y, float(artist_bbox.y1) + pad_pixels)

    display_bbox = Bbox.from_bounds(min_x, min_y, max_x - min_x, max_y - min_y)
    return display_bbox.transformed(fig.dpi_scale_trans.inverted())


def compute_height_offset(
    skeleton: Any,
    q_frames: Sequence[np.ndarray],
    mesh_specs: Sequence[MeshSpec],
) -> float:
    """Minimum world Y over all frames, used to place the ground at y=0."""
    min_y = np.inf
    for q_frame in q_frames:
        vertices_list, _ = rajagopal_world_meshes(skeleton, q_frame, mesh_specs)
        for vertices in vertices_list:
            min_y = min(min_y, float(np.min(vertices[:, 1])))
    if not np.isfinite(min_y):
        return 0.0
    return float(min_y)


def compute_root_offsets(
    skeleton: Any,
    q_frame: np.ndarray,
) -> Tuple[float, float]:
    skeleton.setPositions(np.asarray(q_frame, dtype=np.float64).reshape(-1))
    skeleton.computeForwardKinematics()
    return _pelvis_world_xy(skeleton)
