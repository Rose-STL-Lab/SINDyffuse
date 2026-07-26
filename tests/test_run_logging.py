"""Tests for run logging helpers and script CLI wiring."""

from __future__ import annotations

import argparse
import ast
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class RunLoggingTests(unittest.TestCase):
    def test_build_run_log_paths_accepts_str(self) -> None:
        from common.run_logging import build_run_log_paths

        with tempfile.TemporaryDirectory() as td:
            paths = build_run_log_paths(td, script_name="preprocess_mint")
            self.assertTrue(paths.log_dir.is_dir())
            self.assertEqual(paths.log_dir, Path(td).resolve())
            self.assertTrue(str(paths.log_file).endswith(".log"))

    def test_build_run_log_paths_rejects_namespace(self) -> None:
        from common.run_logging import build_run_log_paths

        ns = argparse.Namespace(log_dir="/tmp")
        with self.assertRaises(TypeError):
            build_run_log_paths(ns, script_name="preprocess_mint")

    def test_dual_tqdm_context_manager(self) -> None:
        from common.run_logging import dual_tqdm, null_logger

        with dual_tqdm(total=2, desc="test", logger=null_logger()) as pbar:
            pbar.update(1)
        # close() called on exit; no exception means success

    def test_preprocess_mint_run_preprocess_empty(self) -> None:
        import scripts.preprocess_mint as pm
        from common.run_logging import null_logger

        with tempfile.TemporaryDirectory() as td:
            hml = Path(td) / "hml"
            hml.mkdir()
            (hml / "train.txt").write_text("", encoding="utf-8")
            (hml / "val.txt").write_text("", encoding="utf-8")
            (hml / "test.txt").write_text("", encoding="utf-8")
            args = argparse.Namespace(
                hml_root=str(hml),
                out_root=str(hml),
                mint_root=str(hml),
                max_motions=0,
                skip_existing=False,
                skip_normalization=True,
                fps=20.0,
                joint_source="auto",
                joints_root="",
                num_shards=1,
                shard_index=0,
                num_workers=1,
                opensim_log_level="Off",
                _run_log_file="",
            )
            with unittest.mock.patch.object(pm, "_check_runtime_dependencies"):
                pm.run_preprocess(args, null_logger())

    def test_run_logged_main_uses_log_dir(self) -> None:
        from common.run_logging import run_logged_main

        seen: list[str] = []

        with tempfile.TemporaryDirectory() as td:
            run_logged_main(
                "unit_test",
                td,
                lambda logger: seen.append(str(logger)),
                argv=["test_run_logging.py"],
            )
            self.assertEqual(len(seen), 1)
            self.assertTrue(any(p.suffix == ".log" for p in Path(td).iterdir()))

    def test_no_script_passes_namespace_to_run_log_session(self) -> None:
        """Static guard: run_log_session(..., args, ...) without .log_dir is a bug."""
        offenders: list[str] = []
        for path in sorted((_REPO / "scripts").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name != "run_log_session":
                    continue
                if not node.args:
                    continue
                first = node.args[0]
                if isinstance(first, ast.Name) and first.id == "args":
                    offenders.append(f"{path.relative_to(_REPO)}:{node.lineno}")
        self.assertEqual(offenders, [])


class EnvironmentConstraintTests(unittest.TestCase):
    def test_scipy_pinned_for_numpy_125(self) -> None:
        import re

        text = (_REPO / "env" / "environment.yaml").read_text(encoding="utf-8")
        self.assertIn("numpy>=1.25,<1.26", text)
        self.assertRegex(text, r"scipy>=1\.11,<1\.14")

    def test_scipy_import_compatible_with_numpy_125(self) -> None:
        import numpy as np
        import scipy

        self.assertTrue(np.__version__.startswith("1.25"), np.__version__)
        major_minor = tuple(int(x) for x in scipy.__version__.split(".")[:2])
        self.assertLess(
            major_minor,
            (1, 14),
            (
                f"SciPy {scipy.__version__} requires numpy>=1.26; OpenSim needs numpy 1.25.x. "
                "Rebuild: mamba env update -f env/environment.yaml --prune"
            ),
        )


class PreprocessMintExportTests(unittest.TestCase):
    def test_export_motion_calls_load_hml3d_joint_positions(self) -> None:
        import scripts.preprocess_mint as pm

        fake_joints = np.random.randn(10, 22, 3).astype(np.float64)
        with tempfile.TemporaryDirectory() as td:
            hml = Path(td) / "hml"
            hml.mkdir()
            out = hml / "mint_cache"
            out.mkdir()
            with unittest.mock.patch.object(
                pm,
                "load_hml3d_joint_positions",
                return_value=(fake_joints, "joints"),
            ) as mock_load:
                with unittest.mock.patch.object(
                    pm,
                    "retarget_hml_joints_to_q",
                    side_effect=RuntimeError("stop-after-load"),
                ):
                    row = pm.export_motion_to_npz(
                        "000001",
                        hml_root=hml,
                        out_root=hml,
                        mint_root=str(hml),
                        skip_existing=False,
                    )
            mock_load.assert_called_once()
            call_args, call_kwargs = mock_load.call_args
            self.assertEqual(call_args[0], hml)
            self.assertEqual(call_args[1], "000001")
            self.assertEqual(row["status"], "error")
            self.assertIn("stop-after-load", row.get("error", ""))


class PreprocessMintMainTests(unittest.TestCase):
    def test_main_logging_setup_does_not_pass_namespace(self) -> None:
        import scripts.preprocess_mint as pm

        with tempfile.TemporaryDirectory() as td:
            argv = [
                "preprocess_mint.py",
                "--max_motions",
                "0",
                "--no_run_log",
                "--log_dir",
                td,
            ]
            with unittest.mock.patch.object(sys, "argv", argv):
                with unittest.mock.patch.object(pm, "run_preprocess") as mock_run:
                    pm.main()
                    mock_run.assert_called_once()
                    logger = mock_run.call_args[0][1]
                    self.assertIsNotNone(logger)


if __name__ == "__main__":
    unittest.main()
