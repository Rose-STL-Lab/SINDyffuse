#!/usr/bin/env python3
"""Deprecated alias — use scripts/build_lai_opensimad_ext.py."""
from __future__ import annotations
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parent / 'build_lai_opensimad_ext.py'), run_name='__main__')
