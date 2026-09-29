from __future__ import annotations
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.modules.setdefault('torch', types.ModuleType('torch'))
sys.modules.setdefault('casadi', types.ModuleType('casadi'))
opensim = sys.modules.setdefault('opensim', types.ModuleType('opensim'))
opensim.Logger = type('Logger', (), {'setLevelString': staticmethod(lambda *_: None), 'getLevelString': staticmethod(lambda: 'Off')})

from common.preprocess_runner import load_manifest_rows, run_preprocess_loop
from common.run_logging import null_logger


class PreprocessResumeTest(unittest.TestCase):
    def test_resume_keeps_success_and_retries_failed_or_partial_rows(self) -> None:
        with tempfile.TemporaryDirectory() as root_s:
            manifest = Path(root_s) / 'manifest.jsonl'
            manifest.write_text(
                json.dumps({'id': 'done', 'status': 'ok'}) + '\n'
                + json.dumps({'id': 'retry', 'status': 'error'}) + '\n'
                + '{"id":"partial"',
                encoding='utf-8',
            )
            calls: list[str] = []

            def process(item: tuple) -> dict:
                calls.append(str(item[0]))
                return {'id': str(item[0]), 'status': 'ok'}

            counts = run_preprocess_loop(
                work=[('done',), ('retry',), ('new',)],
                process_one=process,
                manifest_file=manifest,
                motion_workers=1,
                moco_threads=1,
                ok_statuses={'ok'},
                logger=null_logger(),
            )
            self.assertEqual(calls, ['retry', 'new'])
            self.assertEqual(counts, (3, 0, 0))
            rows = load_manifest_rows(manifest)
            self.assertEqual(set(rows), {'done', 'retry', 'new'})
            self.assertTrue(all(row['status'] == 'ok' for row in rows.values()))


if __name__ == '__main__':
    unittest.main()