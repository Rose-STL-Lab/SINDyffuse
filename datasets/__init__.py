from __future__ import annotations

from typing import TYPE_CHECKING

from common.paths import (
    default_datasets_dir,
    default_humanml3d_root,
    repo_root,
    resolve_data_root,
)

__all__ = [
    "default_datasets_dir",
    "default_humanml3d_root",
    "repo_root",
    "resolve_data_root",
]


if TYPE_CHECKING:
    pass
