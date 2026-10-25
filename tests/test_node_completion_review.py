"""Completion-time comparisons must preserve sequential BT eligibility."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import notebook_qwen_compare_utils as review


def episode(name, human, model, revision=1, family="qwen38"):
    row = {
        "episode_dir": name, "reviewer_sheet_available": True,
        f"{family}_v{revision}_observation_count": 3,
    }
    for slot, (human_time, model_time) in enumerate(zip(human, model), 1):
        row.update({
            f"node_{slot}_node_id": f"node_action_{slot}",
            f"node_{slot}_reviewer_completion_time": human_time,
            f"node_{slot}_{family}_v{revision}_model_completion_time": model_time,
        })
    return row


class CompletionReviewTests(unittest.TestCase):
    def test_correct_case_exploration_preserves_errors_and_denominators(self):
        rows = [
            episode("cc", ["0:00"], ["0:00"], revision=2, family="luna6"),
            episode("ci", [""], [""], revision=2, family="luna6"),
            episode("fc", [""], ["0:01"], revision=2, family="luna6"),
            episode("fi", ["0:01", "0:02"], ["", ""], revision=2, family="luna6"),
        ]
        observations = pd.DataFrame([
            {"episode_dir": name, "model_name": "luna6_v2", "node_id": "node_action_1",
             "node_type": "action", "source": "mission_engine_response", "video_time_s": time,
             "review_row": time + 1, "model_status": status, "model_notes": f"{name}-{time}"}
            for name in ("cc", "ci", "fc", "fi")
            for time, status in ((0, "running"), (1, "complete" if name in ("cc", "fc") else "running"))
        ])
        frame = pd.DataFrame(rows)
        errors = review.build_completion_error_analysis(frame, observations, "luna6_v2")
        all_cases = review.build_completion_error_analysis(frame, observations, "luna6_v2", include_correct=True)
        self.assertEqual(set(all_cases["cases"]["category"]), set(review.COMPLETION_CATEGORIES))
        self.assertEqual(len(all_cases["cases"]), 4)
        self.assertEqual(len(all_cases["observations"]), 8)
        self.assertFalse(all_cases["cases"]["case_id"].duplicated().any())
        self.assertTrue(all_cases["cases"]["node_slot"].eq(1).all())
        for key in ("by_slot", "by_task", "by_action", "by_node"):
            pd.testing.assert_frame_equal(errors[key], all_cases[key])
        pd.testing.assert_frame_equal(
            errors["cases"].reset_index(drop=True),
            all_cases["cases"].loc[all_cases["cases"]["is_error"]].reset_index(drop=True),
        )
        correct = all_cases["cases"].set_index("episode_dir")
        self.assertEqual(correct.loc["cc", "evidence_notes"], "cc-1")
        self.assertEqual(correct.loc["ci", "evidence_notes"], "ci-1")
        self.assertTrue(correct.loc["cc", "case_id"].startswith("C"))

    def test_case_windows_are_pinned_to_batch_and_validate_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ep = root / "episode"
            out = ep / "active"
            out.mkdir(parents=True)
            batch = root / "batch"
            (batch / "pass_01").mkdir(parents=True)
            (batch / "experiment.json").write_text(json.dumps({"prompt": {"sha256": "active"}}))
            (batch / "pass_01/details.jsonl").write_text(json.dumps({
                "episode_dir": str(ep), "qwen_output_dir": str(out),
            }))
            (ep / "episode_summary.json").write_text(json.dumps({"outer_step_seconds": 0.4}))
            (out / "replay_config.json").write_text(json.dumps({"status_prompt_sha256": "active"}))
            frames = [ep / f"frame_{i:05d}.jpg" for i in (1, 2)]
            for path in frames:
                path.touch()
            request = {
                "direction": "request", "node_id": "node_action_grasp", "step": 2,
                "status_eval_id": "same-id-across-attempts", "status_prompt_sha256": "active",
                "frames": [str(path) for path in frames], "request": {"prompt_text": "Exact prompt"},
            }
            response = {
                **request, "direction": "response",
                "response": {"ok": True, "text": '{"status":"complete"}'},
            }
            log = out / "missionengine.jsonl"
            log.write_text("\n".join(json.dumps(row) for row in (request, response)))
            archive = ep / "archived"
            archive.mkdir()
            (archive / "missionengine.jsonl").write_text('{"must_not_be_read":true}')
            result = review.load_completion_case_windows(batch, ep.name, "node_action_grasp")
            self.assertEqual(result["prompt_text"].tolist(), ["Exact prompt"])
            self.assertEqual(result["simulation_time_s"].tolist(), [0.8])
            self.assertEqual(result["first_to_last_simulation_seconds"].tolist(), [0.4])
            self.assertEqual(result["frames"].iloc[0], [str(path) for path in frames])
            for rows in ((request,), (request, request, response),
                         (request, {**response, "status_prompt_sha256": "archived"})):
                log.write_text("\n".join(json.dumps(row) for row in rows))
                with self.assertRaises(ValueError):
                    review.load_completion_case_windows(batch, ep.name, "node_action_grasp")
            log.write_text("\n".join(json.dumps(row) for row in (request, response)))
            frames[0].unlink()
            with self.assertRaises(FileNotFoundError):
                review.load_completion_case_windows(batch, ep.name, "node_action_grasp")

    def test_luna_categories_eligibility_and_revision_selection(self):
        rows = [
            episode("correct_complete", ["0:00"] * 3, ["0:00"] * 3, family="luna6"),
            episode("correct_incomplete", [""] * 3, [""] * 3, family="luna6"),
            episode("false_complete", [""] * 3, ["0:01"] * 3, family="luna6"),
            episode("false_incomplete", ["0:01"] * 3, [""] * 3, family="luna6"),
            episode("missing_luna", ["0:01"] * 3, ["0:01"] * 3),
        ]
        frame = pd.DataFrame(rows)
        result = review.build_node_completion_review(frame, False, qwen38_versions=[])
        counts = result["counts"].set_index("node_slot")
        self.assertEqual(set(counts["model"]), {"Luna 6 v1"})
        for category in review.COMPLETION_CATEGORIES:
            self.assertEqual(counts.loc[1, category], 1)
        self.assertEqual(counts["eligible_count"].tolist(), [4, 2, 2])
        self.assertEqual(result["coverage"]["missing_model"].tolist(), [1, 1, 1])
        detail = result["detail"].set_index(["episode_dir", "node_slot"])
        self.assertEqual(detail.loc[("false_incomplete", 2), "exclusion_reason"], "previous_node_incomplete")
        self.assertTrue(detail.loc[("false_complete", 3), "eligible"])
        qwen_only = review.build_node_completion_review(frame, False, luna_versions=[])
        self.assertEqual(set(qwen_only["counts"]["model"]), {"Qwen3.8 v1"})
        for versions in ([0], [6], [1, 1]):
            with self.subTest(versions=versions), self.assertRaises(ValueError):
                review.build_node_completion_review(frame, luna_versions=versions)

    def test_truth_table_zero_times_and_gate(self):
        rows = [
            episode("correct_complete", ["0:00"] * 3, ["0:00"] * 3),
            episode("correct_incomplete", [""] * 3, [""] * 3),
            episode("false_complete", [""] * 3, ["0:01"] * 3),
            episode("false_incomplete", ["0:01"] * 3, [""] * 3),
        ]
        result = review.build_node_completion_review(pd.DataFrame(rows), False)
        counts = result["counts"].set_index("node_slot")
        for category in review.COMPLETION_CATEGORIES:
            self.assertEqual(counts.loc[1, category], 1)
        self.assertEqual(counts["eligible_count"].tolist(), [4, 2, 2])
        self.assertEqual(counts.loc[2, "False complete"], 1)
        self.assertEqual(counts.loc[3, "False complete"], 1)
        detail = result["detail"].set_index(["episode_dir", "node_slot"])
        self.assertFalse(detail.loc[("false_incomplete", 2), "eligible"])
        self.assertEqual(detail.loc[("false_incomplete", 2), "exclusion_reason"], "previous_node_incomplete")
        self.assertTrue((counts[list(review.COMPLETION_CATEGORIES)].sum(axis=1) == counts["eligible_count"]).all())
        figure, axes = review.plot_node_completion_review(result["counts"])
        self.assertEqual(len(axes), 3)
        self.assertEqual([axis.get_title() for axis in axes], [f"Node {i} Completion" for i in (1, 2, 3)])
        plt.close(figure)

    def test_missing_models_nodes_and_human_sheets_are_not_incomplete(self):
        rows = [
            episode("present", [""] * 3, [""] * 3),
            episode("missing_model", [""] * 3, [""] * 3),
            episode("missing_human", [""] * 3, [""] * 3),
            episode("missing_node", [""] * 3, ["0:01"] * 3),
            episode("broken_sequence", [""] * 3, ["", "0:02", "0:03"]),
        ]
        rows[1]["qwen38_v1_observation_count"] = 0
        rows[2]["reviewer_sheet_available"] = False
        rows[3]["node_3_node_id"] = ""
        result = review.build_node_completion_review(pd.DataFrame(rows), False)
        detail = result["detail"].set_index(["episode_dir", "node_slot"])
        self.assertEqual(detail.loc[("missing_model", 1), "exclusion_reason"], "missing_model")
        self.assertEqual(detail.loc[("missing_human", 1), "exclusion_reason"], "missing_human_review")
        self.assertEqual(detail.loc[("missing_node", 3), "exclusion_reason"], "missing_node")
        self.assertFalse(detail.loc[("broken_sequence", 3), "eligible"])
        self.assertEqual(result["counts"].iloc[0]["eligible_count"], 3)
        coverage = result["coverage"]
        reasons = ["missing_model", "missing_human_review", "missing_node", "previous_node_incomplete"]
        self.assertTrue((coverage[reasons].sum(axis=1) + coverage["eligible_count"] == coverage["episode_count"]).all())

    def test_revision_discovery_missing_reference_and_duplicate_safety(self):
        rows = [
            episode("one", ["0:01"], ["0:01"]),
            episode("two", [""], [""], revision=2),
        ]
        result = review.build_node_completion_review(pd.DataFrame(rows))
        self.assertEqual(
            set(result["counts"]["model"]),
            {*review.COMPLETION_REFERENCE_MODELS, "Qwen3.8 v1", "Qwen3.8 v2"},
        )
        counts = result["counts"]
        self.assertEqual(counts.query("model == 'Qwen3.8 v2' and node_slot == 1")["Correct incomplete"].iloc[0], 1)
        self.assertTrue(counts[counts["model"].isin(review.COMPLETION_REFERENCE_MODELS)]["eligible_count"].eq(0).all())
        with self.assertRaises(ValueError):
            review.build_node_completion_review(pd.DataFrame([rows[0], rows[0]]))
        with self.assertRaises(ValueError):
            review.build_node_completion_review(pd.DataFrame(rows), qwen38_versions=[6])
        self.assertTrue(review.build_node_completion_review(pd.DataFrame())["counts"].empty)

    def test_workbook_discovery_and_existing_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            ep = Path(directory) / "Example_episode_1_of_21_tag"
            ep.mkdir()
            workbook = ep / "human_rater_evaluation.xlsx"
            human = pd.DataFrame({
                "node_order": [1, 2, 3],
                "node_id": ["node_action_1", "node_action_2", "node_action_3"],
                "reviewer_completion_time_m:ss": ["0:00", "", "0:04"],
            })
            model = pd.DataFrame({
                "milestone_idx": [1, 2, 3], "node_type": ["action"] * 3,
                "machine_frame": [0, 60, 120], "machine_time_m:ss": ["0:00", "0:02", "0:04"],
                "sim_success": [False] * 3, "reviewer_agree_YN_partial": [""] * 3,
                "reviewer_time_m:ss": [""] * 3, "reviewer_frame": [""] * 3,
                "reviewer_status": [""] * 3, "reviewer_notes": [""] * 3,
                "evidence_refs": [""] * 3,
                "node_id": ["node_action_1", "node_action_2", "node_action_3"],
                "status_eval_id": ["node_action_1:1", "node_action_2:1", "node_action_3:1"],
                "video_time_m:ss": ["0:00", "0:02", "0:04"],
                "model_status": ["complete", "complete", "incomplete"],
                "model_notes": ["done", "done", "not done"], "model_progress": [1, 1, 0.2],
                "model_latency_s": [0.1, 0.1, 0.1],
            })
            with pd.ExcelWriter(workbook) as writer:
                human.to_excel(writer, sheet_name="Sheet1_BT_Node_Review", startrow=7, index=False)
                for sheet in (
                    "qwen38_v1", "qwen38_v2", "qwen38_v5", "qwen38_v6",
                    "luna6_v1", "luna6_v2", "luna6_v5", "luna6_v6", "luna6_v1_smoke",
                    "sheet5_molmo2_8b", "sheet6_gemma4",
                ):
                    model.to_excel(writer, sheet_name=sheet, startrow=7, index=False)
            sheets = review._read_molmo2_review_sheets(workbook)
            self.assertEqual({name for _, _, name in sheets}, {
                "qwen38_v1", "qwen38_v2", "qwen38_v5",
                "luna6_v1", "luna6_v2", "luna6_v5", "molmo2_8b", "gemma4",
            })
            wide = review.build_btaudit2_molmo2_episode1_dataset(directory)
            self.assertEqual(len(wide), 1)
            for revision in (1, 2, 5):
                self.assertEqual(wide.iloc[0][f"qwen38_v{revision}_observation_count"], 3)
                self.assertEqual(wide.iloc[0][f"node_1_qwen38_v{revision}_model_completion_time"], "0:00")
                self.assertEqual(wide.iloc[0][f"node_3_qwen38_v{revision}_model_completion_time"], "")
                self.assertEqual(wide.iloc[0][f"luna6_v{revision}_observation_count"], 3)
                self.assertEqual(wide.iloc[0][f"node_1_luna6_v{revision}_model_completion_time"], "0:00")
                self.assertEqual(wide.iloc[0][f"node_3_luna6_v{revision}_model_completion_time"], "")
            self.assertFalse(wide.iloc[0]["qwen3_review_available"])
            self.assertFalse(wide.iloc[0]["qwen35_review_available"])
            result = review.build_node_completion_review(wide, False)
            counts = result["counts"].set_index(["model", "node_slot"])
            self.assertEqual(counts.loc[("Qwen3.8 v1", 3), "False incomplete"], 1)
            self.assertEqual(counts.loc[("Luna 6 v1", 3), "False incomplete"], 1)
            with pd.ExcelWriter(workbook, mode="a", engine="openpyxl") as writer:
                pd.DataFrame({"bad": [1]}).to_excel(writer, sheet_name="qwen38_v3", index=False)
            with self.assertRaisesRegex(ValueError, "qwen38_v3"):
                review._read_molmo2_review_sheets(workbook)

    def test_malformed_luna_sheet_is_not_silently_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            workbook = Path(directory) / "human_rater_evaluation.xlsx"
            with pd.ExcelWriter(workbook) as writer:
                pd.DataFrame({"bad": [1]}).to_excel(writer, sheet_name="luna6_v1_smoke", index=False)
            self.assertEqual(review._read_molmo2_review_sheets(workbook), [])
            with pd.ExcelWriter(workbook, mode="a", engine="openpyxl") as writer:
                pd.DataFrame({"bad": [1]}).to_excel(writer, sheet_name="luna6_v1", index=False)
            with self.assertRaisesRegex(ValueError, "luna6_v1"):
                review._read_molmo2_review_sheets(workbook)

    def test_luna_error_evidence_uses_actual_responses_and_same_denominator(self):
        frame = pd.DataFrame([episode(
            "one", ["0:03", "0:05", ""], ["", "", ""], family="luna6",
        )])
        observations = pd.DataFrame([{
            "episode_dir": "one", "model_name": "luna6_v1", "node_id": "node_action_1",
            "node_type": "action", "source": "mission_engine_response",
            "video_time_s": 10, "review_row": 1, "model_status": "running",
            "model_notes": "Lift not established", "model_progress": 0.5,
        }])
        result = review.build_completion_error_analysis(frame, observations, "luna6_v1")
        self.assertEqual(len(result["nodes"]), 1)
        self.assertEqual(len(result["cases"]), 1)
        case = result["cases"].iloc[0]
        self.assertEqual(case["category"], "False incomplete")
        self.assertEqual(case["evidence_notes"], "Lift not established")
        self.assertTrue(case["evaluated_at_or_after_human_completion"])
        self.assertEqual(result["by_slot"]["eligible_count"].sum(), 1)
        self.assertEqual(result["observations"]["model_name"].tolist(), ["luna6_v1"])

    def test_error_evidence_filters_milestones_and_preserves_denominators(self):
        row = episode("one", ["", "0:03", ""], ["0:00", "", ""], revision=2)
        row.update({
            "node_1_node_id": "node_action_grasp", "node_2_node_id": "node_action_place",
            "node_1_Node_description": "Grasp the cup", "node_1_success_criteria": "Cup is held",
            "node_1_reviewer_notes": "Cup never lifted", "mission_name": "Cup task",
            "qwen38_v1_observation_count": 4,
        })
        no_responses = episode("two", ["0:01"], [""], revision=2)
        observations = pd.DataFrame([
            {"node_id": "node_action_grasp", "video_time_s": 0, "model_status": "complete", "model_notes": "Held"},
            {"node_id": "node_action_grasp", "video_time_s": 1, "model_status": "complete", "model_notes": "Later"},
            {"node_id": "node_action_place", "video_time_s": 2, "model_status": "failure", "model_notes": "Cannot place"},
        ])
        observations["source"] = "mission_engine_response"
        observations["episode_dir"] = "one"
        observations["model_name"] = "qwen38_v2"
        observations["node_type"] = "action"
        observations["review_row"] = [2, 3, 4]
        milestone = observations.iloc[[0]].copy()
        milestone["source"] = "btstatus_complete_milestone"
        milestone["review_row"] = 1
        milestone["model_notes"] = "Bookkeeping, not evidence"
        observations = pd.concat([milestone, observations], ignore_index=True)
        result = review.build_completion_error_analysis(pd.DataFrame([row, no_responses]), observations)
        cases = result["cases"]
        self.assertEqual(len(cases), 3)
        grasp = cases[cases["node_slot"].eq(1) & cases["episode_dir"].eq("one")].iloc[0]
        self.assertEqual(grasp["evidence_time_s"], 0)
        self.assertEqual(grasp["evidence_notes"], "Held")
        self.assertEqual(grasp["response_count"], 2)
        self.assertEqual(grasp["action_family"], "grasp")
        self.assertEqual(grasp["human_notes"], "Cup never lifted")
        place = cases[cases["node_slot"].eq(2)].iloc[0]
        self.assertEqual(place["failure_response_count"], 1)
        self.assertFalse(place["evaluated_at_or_after_human_completion"])
        self.assertEqual(place["evidence_notes"], "Cannot place")
        self.assertEqual(cases[cases["episode_dir"].eq("two")]["response_count"].iloc[0], 0)
        self.assertEqual(len(result["observations"]), 3)
        self.assertFalse(result["observations"]["source"].eq("btstatus_complete_milestone").any())
        self.assertEqual(result["by_slot"]["eligible_count"].sum(), len(result["nodes"]))
        self.assertEqual(result["by_task"]["error_count"].sum(), 3)
        with self.assertRaises(ValueError):
            review.build_completion_error_analysis(pd.DataFrame([row]), observations, "qwen38_v5")


if __name__ == "__main__":
    unittest.main()
