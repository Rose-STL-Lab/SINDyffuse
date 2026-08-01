from __future__ import annotations
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)