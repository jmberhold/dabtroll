"""Dataset audits must report incomplete provenance without modifying research inputs."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from openpyxl import Workbook
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_btaudit2_dataset import REQUIRED_FILES, audit_dataset, file_sha256
import build_btaudit2_visualization_dataset as visualization


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.episode = self.root / "Task_episode_2_of_21_test"
        self.episode.mkdir()
        for name in REQUIRED_FILES:
            if name.endswith(".json"):
                (self.episode / name).write_text(json.dumps({"episode_index": 2, "root": {}}))
            elif name.endswith(".jsonl"):
                (self.episode / name).write_text("")
            else:
                (self.episode / name).write_bytes(b"fixture")
        frames = self.episode / "frames" / "camera"
        frames.mkdir(parents=True)
        (frames / "frame_00000.jpg").write_bytes(b"image fixture")
        self.sha = hashlib.sha256(b"prompt").hexdigest()
        self.workbook = self.episode / "human_rater_evaluation.xlsx"
        wb = Workbook()
        wb.active.title = "Sheet1_BT_Node_Review"
        wb.active["A1"] = "Preserve this review"
        wb.create_sheet("qwen38_v4")["A6"] = self.sha
        wb.save(self.workbook)
        wb.close()
        output = self.episode / "qwen38_v4_test"
        output.mkdir()
        self.output = output
        (output / "replay_config.json").write_text(json.dumps({
            "review_sheet_name": "qwen38_v4", "status_prompt_sha256": self.sha,
            "status_max_new_tokens": 1024, "model": "fixture",
        }))
        (output / "status_prompt_source.txt").write_text("prompt")
        (output / "missionengine.jsonl").write_text(json.dumps({
            "direction": "response", "request_kind": "status_eval",
            "response_ok": True, "response_text": '{"status":"complete"}',
        }) + "\n")

    def audit(self):
        return audit_dataset(self.root, 2, 21, "qwen38_v4", self.sha, 1024)

    def test_clean_provenance_is_portable_and_read_only(self):
        before = {str(p): file_sha256(p) for p in self.episode.rglob("*") if p.is_file()}
        report = self.audit()
        self.assertEqual(report["issue_episode_count"], 0)
        self.assertEqual(report["clean_responses"], 1)
        self.assertEqual(report["episodes"][0]["episode"], self.episode.name)
        self.assertNotIn(str(self.root), json.dumps(report))
        self.assertEqual(before, {str(p): file_sha256(p) for p in self.episode.rglob("*") if p.is_file()})

    def test_missing_and_corrupt_inputs_are_not_clean(self):
        (self.episode / "bt.json").write_text("broken")
        (self.episode / "start_frame.jpg").unlink()
        (self.output / "missionengine.jsonl").write_text("{bad json}\n")
        report = self.audit()
        issues = report["episodes"][0]["issues"]
        self.assertEqual(report["issue_episode_count"], 1)
        self.assertTrue(any(s.startswith("invalid:bt.json:") for s in issues))
        self.assertIn("missing:start_frame.jpg", issues)
        self.assertTrue(any(s.startswith("invalid_replay:") for s in issues))

    def test_snapshot_and_token_budget_must_match(self):
        (self.output / "status_prompt_source.txt").write_text("changed")
        self.assertEqual(self.audit()["issue_episode_count"], 1)
        (self.output / "status_prompt_source.txt").write_text("prompt")
        self.assertEqual(audit_dataset(self.root, 2, 21, "qwen38_v4", self.sha, 256)["issue_episode_count"], 1)

    def test_portable_manifest_rejects_duplicates_and_escapes(self):
        manifest = self.root / "manifest.txt"
        for text in (str(self.episode), "../outside", self.episode.name + "\n" + self.episode.name):
            manifest.write_text(text + "\n")
            with self.assertRaises(ValueError):
                audit_dataset(self.root, manifest=manifest)
        manifest.write_text(self.episode.name + "\n")
        self.assertEqual(audit_dataset(self.root, manifest=manifest)["episode_count"], 1)
        with self.assertRaises(ValueError):
            audit_dataset(self.root, 20, 2)

    def test_cli_count_mismatch_fails_and_reports_are_not_overwritten(self):
        output = self.root / "audit"
        command = [
            sys.executable, str(ROOT / "scripts/audit_btaudit2_dataset.py"),
            "--dataset-root", str(self.root), "--output-dir", str(output),
            "--expected-count", "2",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        report_hash = file_sha256(output / "audit.json")
        self.assertFalse(json.loads((output / "audit.json").read_text())["count_matches_expected"])
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(file_sha256(output / "audit.json"), report_hash)

    def test_schedule_paths_are_rebased_to_current_dataset_root(self):
        schedule = pd.DataFrame([{
            "scenario_id": "Task", "run_tag": "test", "episode_index": 2,
            "episode_dir": "/another/workstation/" + self.episode.name,
            "schedule_row": 1,
        }])
        base_row = pd.DataFrame([{
            "scenario_id": "Task", "run_tag": "test", "episode_index": 2,
            "episode_dir": self.episode.name, "success": False,
        }])
        with (
            patch.object(visualization, "_read_schedule", return_value=schedule),
            patch.object(visualization.nqcu.EpisodeWorkbookRowBuilder, "build_episode_row", return_value=base_row),
            patch.object(visualization, "_read_model_sheet", return_value=(pd.DataFrame(), "")),
        ):
            episodes, _ = visualization.build_btaudit2_visualization_dataset(
                logs_dir=self.root, schedule_path=self.root / "schedule.xlsx",
            )
        self.assertEqual(episodes.iloc[0]["episode_path"], str(self.episode))
        self.assertTrue(episodes.iloc[0]["human_reviewed"])


if __name__ == "__main__":
    unittest.main()
