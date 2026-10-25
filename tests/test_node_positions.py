import json
from pathlib import Path
import tempfile
import unittest

from scripts.node_positions import ordered_nodes, original_contract, reconcile_positions


def contract(*identifiers):
    return [
        {"node_id": identifier, "node_position": position, "aliases": []}
        for position, identifier in enumerate(identifiers, 1)
    ]


def response(identifier, row=1, frame=0):
    return {"node_id": identifier, "source": "mission_engine_response",
            "review_row": row, "machine_frame": f"frame_{frame:05}"}


class NodePositionTests(unittest.TestCase):
    def test_known_order_preserves_unreached_prefix_and_suffix(self):
        old = {"nodes": contract("old1", "old2", "old3", "old4"), "basis": "recorded_original_BT"}
        result = reconcile_positions(contract("new1", "new2", "new3"), old, [response("old3")], "qwen3")
        self.assertEqual(result["node_count"], 4)
        self.assertEqual(result["bindings"], {"old3": 3})

    def test_inner_alias_and_syntactic_prefix_not_semantic_guessing(self):
        old = {"nodes": contract("outer"), "basis": "recorded_original_BT"}
        old["nodes"][0]["aliases"] = ["action_old"]
        result = reconcile_positions(contract("new"), old, [response("node_action_old")], "qwen3")
        self.assertEqual(result["bindings"], {"node_action_old": 1})

    def test_replay_uses_current_contract_and_ignores_inherited_milestones(self):
        old = {"nodes": contract("legacy1", "legacy2"), "basis": "recorded_original_BT"}
        records = [response("current1"), {**response("legacy2"), "source": "btstatus_complete_milestone"}]
        result = reconcile_positions(contract("current1", "current2", "current3"), old, records, "qwen35")
        self.assertEqual(result["node_count"], 3)
        self.assertEqual(result["bindings"], {"current1": 1})

    def test_authorized_frame_zero_prefix_retains_recorded_full_count(self):
        old = {"nodes": contract("old1", "old2"), "basis": "recorded_original_BT"}
        result = reconcile_positions(contract("new1", "new2", "new3"), old, [response("other")], "qwen3")
        self.assertEqual(result["node_count"], 2)
        self.assertEqual(result["bindings"], {"other": 1})
        self.assertEqual(result["basis"], "recorded_BT_count_and_observed_prefix_order")
        self.assertEqual(old["nodes"][0]["node_id"], "old1")

    def test_unknown_order_is_not_compressed_without_frame_zero_or_full_contract(self):
        for old, records in (
            ({"nodes": contract("old1", "old2")}, [response("other", frame=5)]),
            ({"nodes": []}, [response("other")]),
            ({"nodes": contract("old1", "old2")}, [response("other"), response("old1", row=2)]),
        ):
            result = reconcile_positions(contract("new1", "new2"), old, records, "qwen3")
            self.assertIsNone(result["node_count"])
            self.assertEqual(result["bindings"], {})

    def test_original_hash_checked_on_reused_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "Task_episode_1_of_21_run"
            target = Path(directory) / "Task_episode_2_of_21_run"
            source.mkdir()
            target.mkdir()
            tree = {"root": {"type": "sequence", "children": [
                {"id": "first", "type": "action", "action": {"id": "inner"}},
                {"id": "second", "type": "condition"},
            ]}}
            (source / "missionengine.jsonl").write_text(json.dumps({
                "direction": "response", "request_kind": "bt",
                "response_text": json.dumps(tree),
            }) + "\n")
            (source / "pipeline_trace.jsonl").write_text(json.dumps({"event": "bt_ready", "bt_hash": "abc"}) + "\n")
            events = [
                {"event": "episode_start", "btaudit_reuse_bt": True, "btaudit_source_episode": 1},
                {"event": "bt_ready", "bt_hash": "abc"},
            ]
            trace = target / "pipeline_trace.jsonl"
            trace.write_text("\n".join(map(json.dumps, events)))
            result = original_contract(target, {})
            self.assertEqual(result["nodes"], ordered_nodes(tree))
            self.assertEqual(result["bt_hash"], "abc")
            events[-1]["bt_hash"] = "different"
            trace.write_text("\n".join(map(json.dumps, events)))
            self.assertEqual(original_contract(target, {})["nodes"], [])


if __name__ == "__main__":
    unittest.main()
