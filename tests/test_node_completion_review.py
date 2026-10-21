"""Completion-time comparisons must preserve sequential BT eligibility."""

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


def episode(name, human, model, revision=1):
    row = {
        "episode_dir": name, "reviewer_sheet_available": True,
        f"qwen38_v{revision}_observation_count": 3,
    }
    for slot, (human_time, model_time) in enumerate(zip(human, model), 1):
        row.update({
            f"node_{slot}_node_id": f"node_action_{slot}",
            f"node_{slot}_reviewer_completion_time": human_time,
            f"node_{slot}_qwen38_v{revision}_model_completion_time": model_time,
        })
    return row


class CompletionReviewTests(unittest.TestCase):
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
                for sheet in ("qwen38_v1", "qwen38_v2", "qwen38_v5", "qwen38_v6", "sheet5_molmo2_8b", "sheet6_gemma4"):
                    model.to_excel(writer, sheet_name=sheet, startrow=7, index=False)
            sheets = review._read_molmo2_review_sheets(workbook)
            self.assertEqual({name for _, _, name in sheets}, {"qwen38_v1", "qwen38_v2", "qwen38_v5", "molmo2_8b", "gemma4"})
            wide = review.build_btaudit2_molmo2_episode1_dataset(directory)
            self.assertEqual(len(wide), 1)
            for revision in (1, 2, 5):
                self.assertEqual(wide.iloc[0][f"qwen38_v{revision}_observation_count"], 3)
                self.assertEqual(wide.iloc[0][f"node_1_qwen38_v{revision}_model_completion_time"], "0:00")
                self.assertEqual(wide.iloc[0][f"node_3_qwen38_v{revision}_model_completion_time"], "")
            self.assertFalse(wide.iloc[0]["qwen3_review_available"])
            self.assertFalse(wide.iloc[0]["qwen35_review_available"])
            result = review.build_node_completion_review(wide, False)
            counts = result["counts"].set_index(["model", "node_slot"])
            self.assertEqual(counts.loc[("Qwen3.8 v1", 3), "False incomplete"], 1)
            with pd.ExcelWriter(workbook, mode="a", engine="openpyxl") as writer:
                pd.DataFrame({"bad": [1]}).to_excel(writer, sheet_name="qwen38_v3", index=False)
            with self.assertRaisesRegex(ValueError, "qwen38_v3"):
                review._read_molmo2_review_sheets(workbook)

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
