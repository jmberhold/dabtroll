"""Batch launchers must not confuse experiments or report failed repairs as clean."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts import dabtroll_bt_pipeline as pipeline
from scripts.prompts import STATUS_PROMPT_FIELDS
from scripts.replay_experiment_identity import experiment_identity


ROOT = Path(__file__).resolve().parents[1]
FAKE_REPLAY = """
import argparse,json,os
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--episode-dir')
args,extra=parser.parse_known_args()
with Path(os.environ['TEST_CALLS']).open('a') as handle:
    handle.write(json.dumps({'episode':args.episode_dir,'args':extra})+'\\n')
mode=os.environ.get('TEST_MODE','success')
if mode=='invalid':
    print('not a replay result')
elif mode=='exit':
    raise SystemExit(7)
else:
    print(json.dumps({'episodes':[{
        'qwen_output_dir':str(Path(args.episode_dir)/'output'),
        'mission_response_rows':1,'mission_response_error_rows':0,'mission_cuda_error_rows':0,
        'mission_has_errors':mode=='mission_error','review_sheet_written':True,
        'mission_error_reason':'test_failure' if mode=='mission_error' else ''
    }]}))
"""


class BatchContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "btaudit2").mkdir()
        self.episodes = []
        for index in range(2, 6):
            episode = self.root / f"Task_episode_{index}_of_21_run"
            episode.mkdir()
            for name in ("episode_summary.json", "bt.json", "status_window_manifest.jsonl",
                         "human_rater_evaluation.xlsx"):
                (episode / name).touch()
            self.episodes.append(str(episode))
        self.manifest = self.root / "episodes.txt"
        self.manifest.write_text("\n".join(self.episodes) + "\n")
        self.fake = self.root / "fake.py"
        self.fake.write_text(FAKE_REPLAY)
        self.calls = self.root / "calls.jsonl"
        self.state = self.root / "state.jsonl"
        self.counter = 0

    def launch(self, script, **changes):
        self.counter += 1
        env = dict(os.environ)
        for key in ("STATUS_PROMPT", "REVIEW_SHEET_NAME", "MODEL_LABEL", "OUTPUT_PREFIX",
                    "STATUS_EVAL_SECONDS", "STATUS_WINDOW_SECONDS", "HISTORY_RUNS_ROOT"):
            env.pop(key, None)
        env.update({
            "PYTHON_BIN": sys.executable, "REPLAY_SCRIPT": str(self.fake),
            "LOGS_ROOT": str(self.root), "SOURCE_MANIFEST": str(self.manifest),
            "STATE_JSONL": str(self.state), "TEST_CALLS": str(self.calls),
            "MAX_CONSEC_FAILS": "20", "MAX_TOTAL_FAILS": "20", "BATCH_SIZE": "10",
            **changes,
        })
        tag = f"run{self.counter}"
        result = subprocess.run(["bash", str(ROOT / "scripts" / script), tag],
                                env=env, text=True, capture_output=True, timeout=30)
        return result, tag

    def count_calls(self):
        return len(self.calls.read_text().splitlines()) if self.calls.exists() else 0

    def test_repair_parse_failures_respect_both_stop_limits(self):
        for limits, expected in (({"MAX_CONSEC_FAILS": "2"}, 2), ({"MAX_TOTAL_FAILS": "3"}, 3)):
            with self.subTest(limits=limits):
                before = self.count_calls()
                result, tag = self.launch("run_replay_btaudit2_sheet2_repair.sh",
                                          TEST_MODE="invalid", **limits)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads((self.root / "btaudit2_sheet2_repair_runs" / tag / "run_report.json").read_text())
                self.assertFalse(report["ok"])
                self.assertTrue(report["stopped_early"])
                self.assertEqual(self.count_calls() - before, expected)
                self.assertEqual(report["remaining_count"], 4)

    def test_repair_failures_are_nonzero_even_without_early_stop(self):
        for mode in ("exit", "mission_error"):
            with self.subTest(mode=mode):
                result, tag = self.launch("run_replay_btaudit2_sheet2_repair.sh", TEST_MODE=mode)
                report = json.loads((self.root / "btaudit2_sheet2_repair_runs" / tag / "run_report.json").read_text())
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(report["ok"])
                self.assertFalse(report["stopped_early"])
                self.assertEqual(report["remaining_count"], 4)

    def test_repair_resume_requires_matching_experiment_including_prompt_content(self):
        self.state.write_text(json.dumps({"episode_dir": self.episodes[0], "ok": True}) + "\n")
        result, _ = self.launch("run_replay_btaudit2_sheet2_repair.sh")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.count_calls(), 4)
        result, _ = self.launch("run_replay_btaudit2_sheet2_repair.sh")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.count_calls(), 4)
        for changes in ({"MODEL_LABEL": "another-model"}, {"REVIEW_SHEET_NAME": "another-sheet"},
                        {"STATUS_PROMPT": "qwen3_5"}):
            before = self.count_calls()
            result, _ = self.launch("run_replay_btaudit2_sheet2_repair.sh", **changes)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(self.count_calls() - before, 4)
        template = self.root / "prompt.txt"
        template.write_text(" ".join("${" + field + "}" for field in sorted(STATUS_PROMPT_FIELDS)))
        for text in ("", "\nA new revision."):
            with template.open("a") as handle:
                handle.write(text)
            before = self.count_calls()
            result, _ = self.launch("run_replay_btaudit2_sheet2_repair.sh", STATUS_PROMPT=str(template))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(self.count_calls() - before, 4)

    def test_batch_sheet_defaults_and_history_are_experiment_specific(self):
        script = "run_replay_episodes_2_21_mod1_batched.sh"
        for sheet, prompt in (("sheet3_qwen3_5_MissionEngine_Review", "standard"),
                              ("sheet4_qwen3_5_mod1_MissionEngine_Review", "qwen3_5")):
            before = self.count_calls()
            result, _ = self.launch(script, REVIEW_SHEET_NAME=sheet)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(self.count_calls() - before, 4)
            call = json.loads(self.calls.read_text().splitlines()[-1])
            self.assertEqual(call["args"][call["args"].index("--status-prompt") + 1], prompt)
            result, _ = self.launch(script, REVIEW_SHEET_NAME=sheet)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(self.count_calls() - before, 4)
        for changes in ({"MODEL_LABEL": "another-model"}, {"STATUS_PROMPT": "standard"},
                        {"STATUS_EVAL_SECONDS": "1"},
                        {"MODEL_LABEL": "allenai/Molmo2-8B", "STATUS_PROMPT": "molmo2_8b",
                         "REVIEW_SHEET_NAME": "molmo2", "OUTPUT_PREFIX": "molmo2"}):
            before = self.count_calls()
            result, _ = self.launch(script, **changes)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(self.count_calls() - before, 4)

    def test_unidentified_batch_history_is_not_reused(self):
        history = self.root / "replay_episodes_2_21_mod1_runs" / "legacy" / "batch_01"
        history.mkdir(parents=True)
        (history / "report.txt").write_text("".join(
            f"[batch_01:{index}/4] RUN {episode}\n  OK responses=1\n  OUT {episode}/output\n"
            for index, episode in enumerate(self.episodes, 1)
        ))
        result, _ = self.launch("run_replay_episodes_2_21_mod1_batched.sh")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.count_calls(), 4)

    def test_identity_normalizes_timing_and_rejects_invalid_seconds(self):
        args = ("sheet", "standard", "model", "prefix", "frames")
        self.assertEqual(experiment_identity(*args, "2", "4"),
                         experiment_identity(*args, "2.0", "4.0"))
        for invalid in ("0", "-1", "nan", "inf", "not-a-number"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                experiment_identity(*args, invalid, "4")

    def test_non_audit_task_override_and_audit_environment_task_reach_bt_request(self):
        for mode, expected in ((None, "Explicit task"), ("btaudit2", "Environment task")):
            with self.subTest(mode=mode):
                env = Mock()
                env.reset.return_value = ({"image": np.zeros((8, 8, 3), dtype=np.uint8)}, {})
                runtime = Mock()
                runtime.ep.episode_dir = self.root
                runtime.video_wrapper_dir = self.root
                task = Mock(state_keys=[], primary_video_key="image")
                task.infer_default_task.return_value = "Environment task"
                mission = Mock()
                mission.status_prompt.source_text = "prompt"
                mission.preflight.return_value = {"ok": True}
                mission.start_request_nonblocking.side_effect = InterruptedError("captured BT request")
                with (
                    patch.object(pipeline, "EpisodeRuntime", return_value=runtime),
                    patch.object(pipeline, "_build_env", return_value=env),
                    patch.object(pipeline, "TaskEngineClient", return_value=task),
                    patch.object(pipeline, "MissionEngineClient", return_value=mission),
                    patch.object(pipeline, "_save_start_frame", return_value=self.root / "start.jpg"),
                ):
                    with self.assertRaisesRegex(InterruptedError, "captured"):
                        pipeline.run_dabtroll_episode(
                            pipeline.PipelineConfig(env_name="test", user_task="Explicit task", test=mode),
                            kb=Mock(),
                        )
                self.assertEqual(mission.start_request_nonblocking.call_args.args[0]["task_text"], expected)
