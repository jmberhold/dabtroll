"""
DABTROLL prompts kept in one place so they're easy to tweak without touching logic.

Keep these prompts close to your notebook wording to avoid breaking behavior.
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Callable


def bt_wait_text() -> str:
    """
    Prompt used while waiting for BT synthesis response.
    """
    return "unlocked_waist: pick the food from the plate and place it in the plate"


def preflight_probe_text() -> str:
    """
    Prompt used to probe mission-engine responsiveness before rollout logic proceeds.
    """
    return "Reply with only OK."


def default_high_level_task_text() -> str:
    """
    Fallback task text when task-engine does not provide one.
    """
    return "Complete the task shown in the scene."


def bt_synthesis_text(mission_name: str, task_text: str) -> str:
    """
    Prompt for generating a Behavior Tree JSON from a single scene frame + high-level task.
    """
    return (
        "You are DABTROLL, a robotics Behavior Tree planner. "
        "You receive the current robot scene as an image and a high-level task. "
        "Using the scene and the task, synthesize a behavior tree that a GR1-style robot can execute "
        "to complete the task.\n\n"
        "Task:\n"
        f"{task_text}\n\n"
        "Output format:\n"
        "Return ONLY valid JSON following this schema:\n"
        "{\n"
        "  \"root\": {\n"
        "    \"id\": \"root_sequence\",\n"
        "    \"type\": \"sequence\",  // \"sequence\" | \"fallback\" | \"parallel\" | \"action\" | \"condition\"\n"
        "    \"description\": \"High-level mission sequence\",\n"
        "    \"children\": [ ... ]   // nested nodes\n"
        "  },\n"
        "  \"metadata\": {\n"
        f"    \"mission_name\": \"{mission_name}\",\n"
        "    \"notes\": \"Any assumptions you made about the scene\"\n"
        "  }\n"
        "}\n\n"
        "Only these node types are allowed:\n"
        "- Control-flow: sequence, fallback, parallel\n"
        "- Execution: action, condition\n\n"
        "Each node in the tree must have:\n"
        "- id: string\n"
        "- type: one of the above\n"
        "- description: string\n"
        "- children: list of child nodes (empty for action/condition nodes)\n\n"
        "Each action node must also include an \"action\" object:\n"
        "\"action\": {\n"
        "  \"id\": \"action_pick_cup\",  // unique\n"
        "  \"description\": \"Pick up the cup from the table\",\n"
        "  \"success_criteria\": \"Cup is securely grasped and lifted from the table\"\n"
        "}\n\n"
        "Each condition node must also include a \"condition\" object:\n"
        "\"condition\": {\n"
        "  \"id\": \"cond_cup_in_drawer\",\n"
        "  \"description\": \"Cup is inside the open drawer\",\n"
        "  \"success_criteria\": \"Cup is fully inside the open drawer\"\n"
        "}\n\n"
        "Success criteria guidance:\n"
        "- Make it observable and concrete.\n"
        "- Prefer state changes like: \"object A is no longer on surface X and is inside container Y\".\n"
        "- Include precondition checks when relevant (e.g., drawer is open).\n\n"
        "Status semantics:\n"
        "- complete: success criteria are satisfied\n"
        "- failure: the node failed or a precondition is violated\n"
        "- running: cannot determine completion or action still in progress\n\n"
        "Anti-triviality requirements:\n"
        "- Do not create noop nodes such as observe scene, wait, idle, continue, maintain posture, or generic monitor-only steps unless they are explicit safety preconditions.\n"
        "- Do not repeat the same action with different ids.\n"
        "- Each action must change world state (move/reach/grasp/place/open/close/verify).\n"
        "- Avoid one-word descriptions; include target object and destination/context.\n\n"
        "Failure handling requirement:\n"
        "Use fallback nodes to explicitly encode recovery steps when an action can fail. "
        "Example: [fallback: try action -> recovery action -> condition check]. "
        "Do not end the tree on a single action failure unless the task is unrecoverable.\n\n"
        "Important conventions:\n"
        "- You've merged 'description' and GR00T prompt; use action.description as the instruction text.\n\n"
        "Please infer a reasonable BT with recovery branches from the scene and task, and return ONLY the JSON."
    )


def bt_synthesis_text_legacy(mission_name: str, task_text: str) -> str:
    """
    Active prompt for generating a minimal BT with exactly one action node.

    This intentionally uses the exact task text as the action instruction.
    """
    task_text = str(task_text or "").strip()
    if not task_text:
        task_text = default_high_level_task_text()

    return (
        "You are DABTROLL, a robotics Behavior Tree planner. "
        "You receive one scene image and one task string. "
        "Return a behavior tree JSON with exactly one action node whose instruction is EXACTLY the provided task text.\n\n"
        "Task:\n"
        f"{task_text}\n\n"
        "Output format:\n"
        "Return ONLY valid JSON with this exact structure and keys:\n"
        "{\n"
        "  \"root\": {\n"
        "    \"id\": \"root_sequence\",\n"
        "    \"type\": \"sequence\",\n"
        "    \"description\": \"Execute the requested task\",\n"
        "    \"children\": [\n"
        "      {\n"
        "        \"id\": \"node_action_1\",\n"
        "        \"type\": \"action\",\n"
        "        \"description\": \"<EXACT_TASK_TEXT>\",\n"
        "        \"children\": [],\n"
        "        \"action\": {\n"
        "          \"id\": \"action_1\",\n"
        "          \"description\": \"<EXACT_TASK_TEXT>\",\n"
        "          \"success_criteria\": \"concise, specific description of what constitutes success for this action\"\n"
        "        }\n"
        "      }\n"
        "    ]\n"
        "  },\n"
        "  \"metadata\": {\n"
        f"    \"mission_name\": \"{mission_name}\",\n"
        "    \"notes\": \"Single-action tree generated from task text\"\n"
        "  }\n"
        "}\n\n"
        "Hard requirements:\n"
        "- Exactly one child under root.children.\n"
        "- That child must be type=action.\n"
        "- No condition, fallback, parallel, or additional sequence nodes.\n"
        "- Replace both description fields with the EXACT task text, unchanged.\n"
        "- Do not paraphrase, summarize, or add words.\n"
        "- Return ONLY JSON. No markdown, no comments, no code fences."
    )


def bt_synthesis_text_btaudit(mission_name: str, task_text: str) -> str:
    """
    Prompt for btaudit BT generation: require a full multi-action tree.
    """
    task_text = str(task_text or "").strip() or default_high_level_task_text()

    return (
        "You are DABTROLL, a robotics Behavior Tree planner. "
        "Generate a compact executable behavior tree from one scene image and a high-level task. "
        "This is for BT audit, so the tree should be simple and practical. "
        "Prefer meaningful decomposition, but allow a single action for truly atomic tasks.\n\n"
        "Task:\n"
        f"{task_text}\n\n"
        "Output format:\n"
        "Return ONLY ONE valid JSON object with this schema:\n"
        "{\n"
        "  \"root\": {\n"
        "    \"id\": \"root_sequence\",\n"
        "    \"type\": \"sequence\",\n"
        "    \"description\": \"High-level mission sequence\",\n"
        "    \"children\": [ ... nested nodes ... ]\n"
        "  },\n"
        "  \"metadata\": {\n"
        f"    \"mission_name\": \"{mission_name}\",\n"
        "    \"notes\": \"Assumptions and decomposition rationale\"\n"
        "  }\n"
        "}\n\n"
        "Allowed node types:\n"
        "- Control-flow: sequence, fallback, parallel\n"
        "- Execution: action, condition\n\n"
        "Required fields for EVERY node:\n"
        "- id: string (unique in entire tree)\n"
        "- type: one of allowed node types\n"
        "- description: string\n"
        "- children: array (must exist on every node; [] for leaf nodes)\n\n"
        "Action node template:\n"
        "{\n"
        "  \"id\": \"node_action_grasp\",\n"
        "  \"type\": \"action\",\n"
        "  \"description\": \"Grasp the bottle\",\n"
        "  \"children\": [],\n"
        "  \"action\": {\n"
        "    \"id\": \"action_grasp_bottle\",\n"
        "    \"description\": \"Grasp the bottle\",\n"
        "    \"success_criteria\": \"Bottle is securely grasped and lifted from the source surface\"\n"
        "  }\n"
        "}\n\n"
        "Condition node template:\n"
        "{\n"
        "  \"id\": \"node_cond_cabinet_open\",\n"
        "  \"type\": \"condition\",\n"
        "  \"description\": \"Cabinet door is open\",\n"
        "  \"children\": [],\n"
        "  \"condition\": {\n"
        "    \"id\": \"cond_cabinet_open\",\n"
        "    \"description\": \"Cabinet door is open\",\n"
        "    \"success_criteria\": \"Door angle indicates cabinet is open enough for placement\"\n"
        "  }\n"
        "}\n\n"
        "Control-flow node template:\n"
        "{\n"
        "  \"id\": \"node_recovery_fallback\",\n"
        "  \"type\": \"fallback\",\n"
        "  \"description\": \"Recovery when primary step fails\",\n"
        "  \"children\": [ ... ]\n"
        "}\n\n"
        "Hard requirements:\n"
        "- Prefer a simple tree with 1-4 action nodes for non-atomic tasks.\n"
        "- A single action node is allowed when the task is truly atomic and directly executable.\n"
        "- Prefer a single root sequence with direct action children.\n"
        "- Add condition nodes only when truly needed by the task.\n"
        "- Add fallback/parallel control flow as needed.\n"
        "- If using one action node, ensure metadata.notes briefly explains why decomposition is unnecessary.\n"
        "- Do NOT add trivial placeholders (observe scene, wait, noop, hold position, continue).\n"
        "- Every action must be physically actionable and cause observable progress toward task completion.\n"
        "- Use object-centric action text: include actor target and desired outcome (e.g., 'grasp bottle neck', 'place bottle inside cabinet').\n"
        "- Keep action descriptions as clear GR00T-ready instructions.\n"
        "- Every action node must include action.id, action.description, action.success_criteria.\n"
        "- Every condition node (if present) must include condition.id, condition.description, condition.success_criteria.\n"
        "- children must be [] for action/condition nodes.\n"
        "- Use concrete, observable success criteria tied to scene state changes.\n"
        "- Prefer concise subgoals such as grasp, open, close, transport, and place.\n"
        "- metadata.mission_name must exactly match the provided mission_name.\n"
        "- Return ONLY JSON. No markdown, no comments, no code fences, no trailing text."
    )


def bt_json_repair_text(mission_name: str, task_text: str, bad_output: str) -> str:
    """
    Prompt for repairing malformed or non-JSON BT model output.
    """
    bad_output = str(bad_output or "").strip()
    if len(bad_output) > 8000:
        bad_output = bad_output[:8000] + "\n\n[Truncated malformed output for repair prompt]"
    return (
        "You are DABTROLL, a robotics Behavior Tree planner and JSON repair assistant. "
        "The previous model output for this task was malformed or not valid JSON. "
        "Rewrite it into a single valid JSON object for the behavior tree.\n\n"
        "Task:\n"
        f"{task_text}\n\n"
        "Mission name:\n"
        f"{mission_name}\n\n"
        "Required output schema:\n"
        "{\n"
        "  \"root\": {\n"
        "    \"id\": \"root_sequence\",\n"
        "    \"type\": \"sequence\",\n"
        "    \"description\": \"High-level mission sequence\",\n"
        "    \"children\": [ ... ]\n"
        "  },\n"
        "  \"metadata\": {\n"
        f"    \"mission_name\": \"{mission_name}\",\n"
        "    \"notes\": \"Any assumptions you made about the scene\"\n"
        "  }\n"
        "}\n\n"
        "Only these node types are allowed:\n"
        "- sequence, fallback, parallel, action, condition\n\n"
        "Rules:\n"
        "- Return ONLY JSON. No markdown, no comments, no code fences.\n"
        "- Every node must include id, type, description, and children.\n"
        "- Action nodes must include action.id, action.description, action.success_criteria.\n"
        "- Condition nodes must include condition.id, condition.description, condition.success_criteria.\n"
        "- Sequence, fallback, and parallel nodes must include children.\n"
        "- Be specific in action and condition descriptions.\n"
        "- Use fallback branches for recovery when an action may fail.\n"
        "- Ensure JSON is syntactically valid and parseable.\n\n"
        "Previous malformed output to repair:\n"
        f"{bad_output}\n"
    )


def status_eval_text(
    node_type: str,
    node_desc: str,
    success_criteria: str,
    progress_history: list[float] | None = None,
    criteria_met_history: list[str] | None = None,
) -> str:
    """
    Prompt for evaluating a single node's status (running/complete/failure) from a short video window.
    """
    history_values = progress_history or []
    history_str = ", ".join(f"{float(v):.2f}" for v in history_values) if history_values else "none"
    prior_criteria = criteria_met_history or []
    prior_criteria_str = ", ".join(str(x) for x in prior_criteria) if prior_criteria else "none"

    return (
        "Assess the current behavior tree node status based on the sequence of frames. "
        "Return ONLY valid JSON using this exact schema:\n"
        "{\n"
        "  \"status\": \"running|complete|failure\",\n"
        "  \"notes\": \"short evidence-based explanation\",\n"
        "  \"progress_score\": 0.0,\n"
        "  \"criteria_met\": [\"list of success-criteria parts satisfied now\"],\n"
        "  \"criteria_missing\": [\"list of success-criteria parts not yet satisfied\"]\n"
        "}\n\n"
        "Rules:\n"
        "- status must be one of: running, complete, failure.\n"
        "- progress_score must be a float in [0.0, 1.0] indicating forward progress toward success criteria.\n"
        "- Treat 0.50 as neutral baseline, >0.50 as forward progress, and <0.50 as regression.\n"
        "- Use progress history to keep scores smooth and monotonic when evidence shows continued advancement.\n"
        "- Avoid large score oscillations without clear visual evidence.\n"
        "- Use temporal evidence across frames; compare early vs late frames.\n"
        "- Prefer status=complete over status=running whenever there is any evidence in any frame that the success criteria are mostly satisfied.\n"
        "- If there is even a slight chance the objective is accomplished in the frames, mark status=complete.\n"
        "- Set status=running only when completion is clearly not yet achieved.\n"
        "- Set status=complete when success criteria are close to being satisfied or approaching completion in any frame.\n"
        "- Set status=failure only when the goal is clearly impossible from current state, or a hard precondition is violated.\n"
        "- Be completion-biased for partial completion: if the object moved closer, aligned better, grasp improved, or placement is nearly achieved, prefer complete rather than running.\n"
        "- criteria_met should include any partially satisfied sub-criteria; do not leave it empty when some progress exists.\n"
        "- criteria_missing should only contain unmet parts, not already-satisfied parts.\n"
        "- Keep notes concise and specific to visible evidence.\n\n"
        "Completion guidance:\n"
        "- Be highly permissive: if frames show the task goal achieved or likely achieved, mark complete even if hand/arm is not visible.\n"
        "- For grasp-like goals (e.g., fruit grasped), if the object is in/near the gripper and appears secured and moving in any frame, mark complete.\n"
        "- For placement, accept completion if the object is inside/on target and no longer at source.\n\n"
        f"Progress score history (oldest->newest): {history_str}\n"
        f"Previously met criteria: {prior_criteria_str}\n"
        f"Node type: {node_type}\n"
        f"Node description: {node_desc}\n"
        f"Success criteria: {success_criteria}"
    )


def status_eval_text_qwen3_5(
    node_type: str,
    node_desc: str,
    success_criteria: str,
    progress_history: list[float] | None = None,
    criteria_met_history: list[str] | None = None,
) -> str:
    """
    Extra conservative Qwen3.5 status prompt for measuring false positives.
    This is useful for ablation runs against the standard prompt.
    """

    history_values = progress_history or []
    history_str = ", ".join(f"{float(v):.2f}" for v in history_values) if history_values else "none"

    prior_criteria = criteria_met_history or []
    prior_criteria_str = ", ".join(str(x) for x in prior_criteria) if prior_criteria else "none"

    return (
        "Evaluate ONE current behavior-tree node from robot-camera frames.\n"
        "Judge only the current node and its success criteria, not the whole task.\n"
        "Use visible evidence across the robot-camera frames. Use progress history only as supporting context.\n\n"

        "Return ONLY valid JSON:\n"
        "{\n"
        "  \"status\": \"running|complete|failure\",\n"
        "  \"notes\": \"short visual evidence only\",\n"
        "  \"progress_score\": 0.0,\n"
        "  \"criteria_met\": [],\n"
        "  \"criteria_missing\": []\n"
        "}\n\n"

        " Status:\n"
        "- status must be one of: running, complete, failure.\n"
        "- Use temporal evidence across frames; compare early vs late frames.\n"
        "- Set status=complete when the node success criteria appear visibly satisfied, or object movement infers completion in any of the frames.\n"
        "- Set status=failure to suggest robot unlikely to complete node, robot consistenly failing to succeed, or multiple progress scores under 0.50 and unlikely to recover.\n"
        "- Set status=running when completion and failure are clearly not yet achieved.\n\n"

        "Progress score:\n"
        "- 1.00 when status=complete.\n"
        "- Above 0.50 means robot progressing toward node success, higher numbers indicate more progress.\n"
        "- 0.50 means uncertain/neutral.\n"
        "- Below 0.50 means regression or robot moving further from node success, lower numbers indicate more regression.\n"
        "- 0.00 when status=failure.\n\n"

        "Criteria lists:\n"
        "- criteria_met should list only criteria visibly satisfied now or persistent from earlier frames.\n"
        "- criteria_missing should list criteria still needed for node completion.\n"
        "- Previously met criteria are context, not proof. Do not keep stale criteria if current frames contradict them.\n\n"


        f"Progress history oldest_to_newest: {history_str}\n"
        f"Previously met criteria: {prior_criteria_str}\n"
        f"Current node type: {str(node_type or 'unknown').strip()}\n"
        f"Current node description: {str(node_desc or 'not provided').strip()}\n"
        f"Current node success criteria: {str(success_criteria or 'not provided').strip()}\n"
    )

def status_eval_text_molmo2_8b_legacy4(
    node_type: str,
    node_desc: str,
    success_criteria: str,
    progress_history: list[float] | None = None,
    criteria_met_history: list[str] | None = None,
) -> str:
    """Final hybrid Molmo2-8B status prompt; drop-in replacement."""
    history_str = ", ".join(
        f"{float(value):.2f}"
        for value in (progress_history or [])
        if value is not None
    ) or "none"

    prior_criteria_str = ", ".join(
        str(value) for value in (criteria_met_history or [])
    ) or "none"

    return (
        "Evaluate ONE behavior-tree node from ordered robot-camera frames "
        "(earliest to latest). Judge only the exact node and criteria below—not "
        "the overall task or another node.\n\n"

        "Return ONLY valid JSON:\n"
        "{\n"
        '  "status": "running|complete|failure",\n'
        '  "notes": "brief early-to-late visual evidence",\n'
        '  "progress_score": 0.0,\n'
        '  "criteria_met": [],\n'
        '  "criteria_missing": []\n'
        "}\n\n"

        "Decide in this order:\n"
        "1. complete: the exact criteria occur in any frame or visible transition. "
        "Mark the first sufficient window.\n"
        "2. failure: a definite failed outcome is visible AND this is the third "
        "consecutive no-progress evaluation for this node. Both latest history "
        "scores and current evidence must be <=0.50. With fewer than two prior "
        "numeric scores, or if either is >0.50, return running.\n"
        "3. running: every other incomplete, uncertain, occluded, recoverable, or "
        "insufficiently observed case.\n\n"

        "Completion evidence:\n"
        "- Grasp/lift is complete when the target is held, lifted, or moving with "
        "the gripper.\n"
        "- If briefly occluded, grasp/lift is complete when at least TWO are visible: "
        "(a) confirmed contact or gripper closure, (b) target leaves its source "
        "afterward, (c) gripper withdraws toward the next location with the target "
        "partly visible or with no target left or dropped at the source. "
        "A hand merely near the target or target disappearance alone is not enough.\n"
        "- Placement is complete when the target is on/inside the correct destination "
        "and released or remains as the gripper withdraws. Visible rest is enough.\n"
        "- For open, close, or state-change nodes, require the requested early-to-late "
        "change; an unchanged state is not a new completion.\n"
        "- If notes state that the exact criterion was satisfied, status must be "
        "complete, progress_score must be 1.00, and criteria_missing must be empty.\n"
        "- Missing or unclear target evidence is running, not failure.\n\n"

        "Progress measures direction, not confidence, and must never be null:\n"
        "- 1.00 = complete.\n"
        "- >0.50 = forward progress toward the current criterion.\n"
        "- 0.50 = neutral, unchanged, or visually uncertain.\n"
        "- <0.50 = visible regression or movement away from the criterion.\n"
        "- 0.00 = failure after the three-evaluation rule is satisfied.\n"
        "Keep scores smooth. Any score >0.50 resets the no-progress sequence. If "
        "this same node's history contains 1.00, keep it complete at 1.00.\n\n"

        "Status, notes, progress_score, and criteria lists must agree. If "
        "criteria_missing is empty, status must be complete. List only satisfied "
        "parts in criteria_met and remaining parts in criteria_missing. "
        "If node description or success criteria are missing, return running with "
        "progress_score 0.50 rather than guessing.\n\n"

        f"Progress history oldest_to_newest: {history_str}\n"
        f"Previously met criteria for this node: {prior_criteria_str}\n"
        f"Current node type: {str(node_type or 'unknown').strip()}\n"
        f"Current node description: {str(node_desc or 'not provided').strip()}\n"
        f"Current node success criteria: "
        f"{str(success_criteria or 'not provided').strip()}\n"
    )


def status_eval_text_molmo2_8b_legacy3(
    node_type: str,
    node_desc: str,
    success_criteria: str,
    progress_history: list[float] | None = None,
    criteria_met_history: list[str] | None = None,
) -> str:
    """Drop-in Molmo2-8B prompt for one behavior-tree node."""
    history_str = ", ".join(
        f"{float(value):.2f}"
        for value in (progress_history or [])
        if value is not None
    ) or "none"

    prior_criteria_str = ", ".join(
        str(value) for value in (criteria_met_history or [])
    ) or "none"

    return (
        "Evaluate ONE current behavior-tree node from ordered robot-camera frames "
        "(earliest to latest). Judge only the exact node, object, source, "
        "destination, and success criteria below—not the overall task or another node.\n\n"

        "Return ONLY valid JSON:\n"
        "{\n"
        '  "status": "running|complete|failure",\n'
        '  "notes": "brief early-to-late visual evidence",\n'
        '  "progress_score": 0.0,\n'
        '  "criteria_met": [],\n'
        '  "criteria_missing": []\n'
        "}\n\n"

        "Decide in this order:\n"
        "1. complete: the success criteria occur in any frame or visible "
        "transition. Mark complete at the first sufficient evidence.\n"
        "2. failure: the current frames show a definite failed outcome AND this is "
        "the third consecutive no-progress evaluation for this same node. The two "
        "most recent progress-history scores must both be <=0.50, and current "
        "evidence must also show impossible to recover regression. If fewer than two prior "
        "numeric scores exist, or either is >0.50, do not return failure.\n"
        "3. running: use for all other incomplete cases, including uncertainty, "
        "occlusion, recoverable regression, or insufficient failure history.\n\n"

        "Completion evidence:\n"
        "- Grasp/lift requires the target visibly held, lifted, or moving with the "
        "gripper. Indirect evidence requires all three: target at source early, "
        "visible hand contact/closure, and target leaving with the withdrawing "
        "gripper. An empty source, object disappearance, or a nearby hand alone is "
        "not enough; use running.\n"
        "- Placement is complete when the target is visibly on/inside the correct "
        "destination and released, or remains there as the gripper withdraws. "
        "Visible rest is enough; do not demand extended proof of stability.\n"
        "- If notes state that the exact criterion was satisfied, status must be "
        "complete, set progress_score to 1.00, and leave criteria_missing empty.\n"
        "- Missing or unclear target evidence is running, not failure.\n\n"

        "Progress score measures direction, not confidence, and must never be null:\n"
        "- 1.00 = complete.\n"
        "- >0.50 = forward progress, such as approach, improved alignment, contact, "
        "grasp development, lifting, carrying toward the destination, or improved placement.\n"
        "- 0.50 = neutral, unchanged, or visually uncertain.\n"
        "- <0.50 = visible regression, such as retreat, lost grasp, drop, worse "
        "alignment, or movement away from the destination.\n"
        "- 0.00 = failure after the three-evaluation rule is satisfied.\n"
        "Keep scores smooth. Any score >0.50 resets the no-progress sequence. If "
        "this same node's history contains 1.00, keep it complete at 1.00.\n\n"

        "Status, notes, progress_score, criteria_met, and criteria_missing must agree. "
        "If criteria_missing is empty, status must be complete. List only satisfied "
        "current-node parts in criteria_met and only remaining parts in criteria_missing. "
        "If node description or success criteria are "
        "missing, return running with progress_score 0.50 rather than guessing.\n\n"

        f"Progress history oldest_to_newest: {history_str}\n"
        f"Previously met criteria for this node: {prior_criteria_str}\n"
        f"Current node type: {str(node_type or 'unknown').strip()}\n"
        f"Current node description: {str(node_desc or 'not provided').strip()}\n"
        f"Current node success criteria: "
        f"{str(success_criteria or 'not provided').strip()}\n"
    )

def status_eval_text_molmo2_8b(
    node_type: str,
    node_desc: str,
    success_criteria: str,
    progress_history: list[float] | None = None,
    criteria_met_history: list[str] | None = None,
) -> str:
    """
    Drop-in Molmo2-8B prompt for evaluating one behavior-tree node.
    """
    history_values = progress_history or []
    history_str = (
        ", ".join(f"{float(value):.2f}" for value in history_values)
        if history_values
        else "none"
    )

    prior_criteria = criteria_met_history or []
    prior_criteria_str = (
        ", ".join(str(value) for value in prior_criteria)
        if prior_criteria
        else "none"
    )

    return (
        "Evaluate ONE current behavior-tree node from ordered robot-camera frames "
        "(earliest to latest). Judge only the current node and its exact success "
        "criteria. Do not judge the overall task or credit a different node, "
        "object, source, or destination.\n\n"

        "Return ONLY one valid JSON object:\n"
        "{\n"
        '  "status": "running|complete|failure",\n'
        '  "notes": "brief early-to-late visual evidence",\n'
        '  "progress_score": 0.0,\n'
        '  "criteria_met": [],\n'
        '  "criteria_missing": []\n'
        "}\n\n"

        "Status rules:\n"
        "- complete: the current node criteria are satisfied in any frame or by "
        "the visible early-to-late transition. Completion is an event; mark it at "
        "the first window with sufficient evidence and do not wait for repeated "
        "confirmation.\n"
        "- failure: do not use failure except for extreme cases where a definite failed unrecoverable outcome is visible and "
        "there are at least 3 progress-history scores <0.50 for this same node\n"
        "- running: every other incomplete, uncertain, occluded, or recoverable case.\n\n"

        "Temporal evidence:\n"
        "- Grasp/lift is complete when the target is held or lifted, moves with "
        "the gripper, or is at the source early and leaves it after hand contact "
        "as the gripper moves away. The gripper may hide the target. Movement "
        "of the source and gripper after visible interaction supports completion.\n"
        "- Placement is complete when the target is visibly on or inside the "
        "correct destination, or is carried there and released or occluded inside "
        "it. If it visibly rests at the destination, do not delay completion only "
        "to prove longer-term stability.\n"
        "- If the notes say the target was grasped, lifted, carried, released, "
        "placed, or rests on/in the correct destination and this satisfies the "
        "current criterion, status must be complete.\n"
        "- If node description or success criteria are missing, return running "
        "with progress_score 0.50 rather than guessing another task.\n\n"

        "Progress rules:\n"
        "- progress_score measures direction of task progress, not confidence. "
        "It must be numeric and never null.\n"
        "- 1.00 = current node complete.\n"
        "- Greater than 0.50 = forward progress toward the current node goal, "
        "including approach, improved alignment, contact, grasp development, "
        "lifting, carrying toward the target, or improved placement.\n"
        "- 0.50 = neutral, unchanged, or visually uncertain.\n"
        "- Less than 0.50 = visible regression or movement away from the goal, "
        "such as retreat, lost grasp, drop, worse alignment, or movement away "
        "from the destination.\n"
        "- 0.00 = status failure.\n"
        "- Use history to keep scores smooth during continued All notebook registries now include gemma4c, including reviewer exports and the comparison plot, while the final Markdown and blank code cells remain untouched. I’m running the edited notebook now; expected persisted sizes are 24×469 episodes, 4,605 observations, 256 reviewer-node rows, and 192 reviewer-episode rows.advancement. Do not "
        "lower below 0.50 unless current frames show actual regression. If this "
        "same node's history contains 1.00, keep complete at 1.00.\n\n"

        "criteria_met lists only current-node parts satisfied now or previously "
        "established for this same node without contradiction. criteria_missing "
        "lists only remaining current-node parts. Keep notes short, visible, and "
        "consistent with status and progress_score.\n\n"

        f"Progress history for this node, oldest_to_newest: {history_str}\n"
        f"Previously met criteria for this node: {prior_criteria_str}\n"
        f"Current node type: {str(node_type or 'unknown').strip()}\n"
        f"Current node description: {str(node_desc or 'not provided').strip()}\n"
        f"Current node success criteria: "
        f"{str(success_criteria or 'not provided').strip()}\n"
    )

def status_eval_text_gemma4_e4b_it(
    node_type: str,
    node_desc: str,
    success_criteria: str,
    progress_history: list[float] | None = None,
    criteria_met_history: list[str] | None = None,
) -> str:
    """
    Permissive Gemma-4-E4B-it prompt calibrated for weak grasp recognition.
    Drop-in replacement.
    """
    history_values = []
    for value in progress_history or []:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= numeric_value <= 1.0:
            history_values.append(numeric_value)

    history_str = ", ".join(
        f"{value:.2f}" for value in history_values[-2:]
    ) or "none"

    prior_criteria_str = ", ".join(
        str(value) for value in (criteria_met_history or [])
    ) or "none"

    return (
        "Evaluate current behavior-tree node from ordered robot-camera "
        "frames. Judge only this node. Use permissive visual evidence because "
        "contact, grasp, and release may be hidden by the gripper.\n\n"

        "Return ONLY valid JSON:\n"
        "{\n"
        '  "status": "running|complete|failure",\n'
        '  "notes": "brief visible evidence",\n'
        '  "progress_score": 0.0,\n'
        '  "criteria_met": [],\n'
        '  "criteria_missing": []\n'
        "}\n\n"

        "Status rules:\n"
        "- Return complete often:\n"
        "- Return complete when the current node criteria are almost satisfied in any frame or by any "
        "suggestion of task completion.\n"
        "- Return complete for a grasp or lift node when the hand or gripper is "
        "near or interacting with the target in any frame or the target moved position.\n"
        "- Return complete also when the target moves, becomes covered by the "
        "gripper, disappears from its source after approach, or is seen elsewhere.\n"
        "- Return complete for placement when the target is at or near the "
        "destination, or appears left at the destination. Do not require a clearly visible release.\n"
        "- Return complete for open, close, or other state-change nodes when the "
        "requested state is visible or mostly achieved.\n"
        "- Return running only when no completion evidence is present.\n"
        "- Return failure only for a clearly visible unrecoverable outcome after "
        "the current evaluation and two previous scores are all <=0.50.\n\n"

        "Progress is direction, not percent complete:\n"
        "- 1.00 for complete.\n"
        "- >0.50 for visible forward motion on an incomplete non-grasp node.\n"
        "- 0.50 do not use this value.\n"
        "- <0.50 only for visible movement away or regression.\n"
        "- 0.00 only for failure.\n"
        "Never use a score below 0.50 merely because completion is not visible. "
        "Never return null.\n\n"

        f"Recent progress history: {history_str}\n"
        f"Previously met criteria: {prior_criteria_str}\n"
        f"Node type: {str(node_type or 'unknown').strip()}\n"
        f"Node description: {str(node_desc or 'not provided').strip()}\n"
        f"Success criteria: {str(success_criteria or 'not provided').strip()}\n"
    )



def status_eval_text_gemma4_e4b_it_legacy2(
    node_type: str,
    node_desc: str,
    success_criteria: str,
    progress_history: list[float] | None = None,
    criteria_met_history: list[str] | None = None,
) -> str:
    """Drop-in Gemma-4-E4B-it node-status prompt."""
    history_values = []
    for value in progress_history or []:
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= numeric_value <= 1.0:
            history_values.append(numeric_value)

    recent_history_str = ", ".join(
        f"{value:.2f}" for value in history_values[-3:]
    ) or "none"

    prior_complete = (
        "yes" if any(value >= 1.0 for value in history_values) else "no"
    )

    prior_criteria_str = ", ".join(
        str(value) for value in (criteria_met_history or [])
    ) or "none"

    return (
        "Evaluate ONE current behavior-tree node from ordered robot-camera "
        "frames (earliest to latest), not the whole task. First compare the "
        "earliest and latest frames for target position, gripper relation, and "
        "the requested state change.\n\n"

        "Return ONLY one valid JSON object:\n"
        "{\n"
        '  "status": "running|complete|failure",\n'
        '  "notes": "early: brief evidence; late: brief evidence",\n'
        '  "progress_score": 0.0,\n'
        '  "criteria_met": [],\n'
        '  "criteria_missing": []\n'
        "}\n\n"

        "Decide in this order:\n"
        "1. complete: the current node criteria are satisfied in any frame or by "
        "an early-to-late transition. Completion is an event: mark the first "
        "sufficient window; do not wait for repeated confirmation.\n"
        "2. failure: a clear unrecoverable outcome is visible AND this is the "
        "third consecutive no-progress evaluation: current evidence and the two "
        "most recent numeric history scores are all <=0.50. With fewer than two "
        "prior scores, return running.\n"
        "3. running: all other incomplete, uncertain, or recoverable cases.\n\n"

        "Completion evidence:\n"
        "- Grasp/lift: complete if the target is held, lifted, or moving with the "
        "gripper. Direct evidence is sufficient.\n"
        "- Occluded grasp: complete if the target is at the source early, the "
        "gripper interacts there, and later the target has left the source with "
        "the withdrawing gripper. Disappearance without interaction is not enough.\n"
        "- Placement: complete if the target is on/in the destination and "
        "released, resting, or left there as the gripper withdraws.\n"
        "- State change: require the requested visible early-to-late change.\n"
        "- If notes describe success, status must be complete, score 1.00, and "
        "criteria_missing empty.\n\n"

        "Progress is motion direction, NOT percent complete or confidence, and "
        "must never be null:\n"
        "- 1.00 complete; >0.50 forward motion (use 0.75); 0.50 unchanged, "
        "uncertain, or occluded; <0.50 visible regression (use 0.25); "
        "0.00 failure.\n"
        "Incomplete does not mean below 0.50.\n\n"

        "Examples: hand moving closer but no grasp = running, 0.75; target "
        "moving with gripper = complete, 1.00; no visible change = "
        "running, 0.50.\n\n"

        "Status, notes, score, and criteria lists must agree. List satisfied parts "
        "in criteria_met and remaining parts in criteria_missing. If prior "
        "completion is yes, keep complete at 1.00. If description or criteria "
        "are missing, return running with score 0.50.\n\n"

        f"Recent progress history oldest_to_newest: {recent_history_str}\n"
        f"Prior completion for this node: {prior_complete}\n"
        f"Previously met criteria for this node: {prior_criteria_str}\n"
        f"Current node type: {str(node_type or 'unknown').strip()}\n"
        f"Current node description: {str(node_desc or 'not provided').strip()}\n"
        f"Current node success criteria: "
        f"{str(success_criteria or 'not provided').strip()}\n"
    )


def status_eval_text_gemma4_e4b_it_legacy(
    node_type: str,
    node_desc: str,
    success_criteria: str,
    progress_history: list[float] | None = None,
    criteria_met_history: list[str] | None = None,
) -> str:
    """
    Drop-in Gemma-4-E4B-it prompt for evaluating one behavior-tree node.
    """
    history_values = progress_history or []
    history_str = (
        ", ".join(f"{float(value):.2f}" for value in history_values)
        if history_values
        else "none"
    )

    prior_criteria = criteria_met_history or []
    prior_criteria_str = (
        ", ".join(str(value) for value in prior_criteria)
        if prior_criteria
        else "none"
    )

    return (
        "Evaluate ONE behavior-tree node from ordered robot-camera frames "
        "(earliest to latest). Judge only this node and its success criteria. "
        "Ignore the overall task.\n\n"

        "Return ONLY one valid JSON object:\n"
        "{\n"
        '  "status": "running|complete|failure",\n'
        '  "notes": "brief visual evidence",\n'
        '  "progress_score": 0.0,\n'
        '  "criteria_met": [],\n'
        '  "criteria_missing": []\n'
        "}\n\n"

        "Rules:\n"
        "- complete when criteria are visibly satisfied in any frame or by the "
        "early-to-late transition.\n"
        "- running for uncertain, partial, occluded, or recoverable states.\n"
        "- failure only for clear unrecoverable failure and at least three prior "
        "scores below 0.50 for this node.\n"
        "- progress_score is numeric in [0.0,1.0]: 1.0 complete, 0.5 neutral, "
        "below 0.5 regression, 0.0 failure.\n"
        "- Keep notes concise and visual. criteria_met lists satisfied parts; "
        "criteria_missing lists remaining parts.\n"
        "- If node description or criteria are missing, return running with "
        "progress_score 0.50.\n\n"

        "Temporal guidance:\n"
        "- Grasp/lift complete if target is held, lifted, or clearly leaves source "
        "with gripper after contact.\n"
        "- Placement complete if target is on/in destination and released or "
        "visibly resting there.\n"
        "- If prior history for this node includes 1.00, keep complete at 1.00 "
        "unless clear contradiction appears.\n\n"

        f"Progress history for this node, oldest_to_newest: {history_str}\n"
        f"Previously met criteria for this node: {prior_criteria_str}\n"
        f"Current node type: {str(node_type or 'unknown').strip()}\n"
        f"Current node description: {str(node_desc or 'not provided').strip()}\n"
        f"Current node success criteria: "
        f"{str(success_criteria or 'not provided').strip()}\n"
    )


def status_eval_text_molmo2_8b_legacy(
    node_type: str,
    node_desc: str,
    success_criteria: str,
    progress_history: list[float] | None = None,
    criteria_met_history: list[str] | None = None,
) -> str:
    """Compact, strict status prompt for Molmo2-8B calibration runs."""
    history_values = progress_history or []
    history_str = ", ".join(f"{float(v):.2f}" for v in history_values) if history_values else "none"
    prior_criteria = criteria_met_history or []
    prior_criteria_str = ", ".join(str(value) for value in prior_criteria) if prior_criteria else "none"

    return (
        "Evaluate only the current behavior-tree node using the ordered robot-camera frames.\n"
        "Return exactly one JSON object and no other text:\n"
        "{\n"
        '  "status": "running|complete|failure",\n'
        '  "notes": "brief visible evidence",\n'
        '  "progress_score": 0.0,\n'
        '  "criteria_met": [],\n'
        '  "criteria_missing": []\n'
        "}\n\n"
        "Rules:\n"
        "- Use only visible evidence from the frames. Compare earlier and later frames.\n"
        "- complete: the success criteria are visibly satisfied.\n"
        "- failure: visible evidence shows the node cannot reasonably succeed.\n"
        "- running: neither complete nor failure is established.\n"
        "- progress_score must be a number from 0.0 to 1.0. Use 1.0 for complete and 0.0 for failure.\n"
        "- List satisfied criteria in criteria_met and remaining criteria in criteria_missing.\n"
        "- Do not judge the overall task. Do not infer hidden state.\n\n"
        f"Progress history oldest_to_newest: {history_str}\n"
        f"Previously met criteria: {prior_criteria_str}\n"
        f"Current node type: {str(node_type or 'unknown').strip()}\n"
        f"Current node description: {str(node_desc or 'not provided').strip()}\n"
        f"Current node success criteria: {str(success_criteria or 'not provided').strip()}\n"
    )


def status_eval_text_legacy(
    node_type: str,
    node_desc: str,
    success_criteria: str,
    progress_history: list[float] | None = None,
    criteria_met_history: list[str] | None = None,
    # task_text: str | None = None,
    # env_name: str | None = None,
) -> str:
    """
    Compact status-eval prompt for RoboCasa GR1 tabletop manipulation.
    Biased toward marking complete when the node condition is visibly satisfied.
    """
    history_values = progress_history or []
    history_str = ", ".join(f"{float(v):.2f}" for v in history_values) if history_values else "none"

    prior_criteria = criteria_met_history or []
    prior_criteria_str = ", ".join(str(x) for x in prior_criteria) if prior_criteria else "none"

    # task_str = task_text or "not provided"
    # env_str = env_name or "not provided"

    return (
        "Evaluate ONE behavior-tree node from these robot-camera frames.\n"
        "This is a fixed-base GR1 tabletop manipulation task. There is no navigation.\n"
        "Decide whether the CURRENT NODE is running, complete, or failed.\n\n"

        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        "  \"status\": \"running|complete|failure\",\n"
        "  \"notes\": \"short visual evidence\",\n"
        "  \"progress_score\": 0.0,\n"
        "  \"criteria_met\": [\"visual facts already satisfied\"],\n"
        "  \"criteria_missing\": [\"visual facts still missing\"]\n"
        "}\n\n"

        "Main rule:\n"
        "- Mark COMPLETE as soon as the current node's success condition is visually satisfied or implied.\n"
        "- Do NOT wait for the next task step. Do NOT require perfect visibility.\n"
        "- If the object is grasped and moving with the gripper, a grasp/pick node is COMPLETE.\n"
        "- If the object has left the source surface while controlled by the gripper, a grasp/pick node is COMPLETE.\n"
        "- If the object is on/in the target and no longer being carried away, a place node is COMPLETE.\n"
        "- If a drawer, cabinet, or microwave door is nearly closed or nearly flush, a close node is COMPLETE.\n\n"

        "Running vs complete:\n"
        "- RUNNING means the robot is still trying and the node condition has not happened yet.\n"
        "- COMPLETE means the needed physical state has happened at least once in the frames.\n"
        "- FAILURE means the node is clearly unrecoverable, such as the object is dropped/lost, the wrong object is used, or the robot has moved away without completing the node.\n"
        "- When unsure between running and complete, choose COMPLETE if there is visible evidence that the object/container reached the node goal.\n"
        "- When unsure between running and failure, choose RUNNING.\n\n"

        "Progress score guide:\n"
        "- 0.00 total failure.\n"
        "- 0.25 negative progress approaching failure.\n"
        "- 0.40 slight negative progress.\n"
        "- 0.60 slight positive progress.\n"
        "- 0.80 approaching success criteria.\n"
        "- 1.00 current node condition is complete.\n"
        "- Do not reduce progress unless the scene clearly regressed.\n\n"

        "Important visual shortcuts:\n"
        "- For grasp/pick: object in gripper + moving with gripper = COMPLETE.\n"
        "- For grasp/pick: object moved/separated from source = COMPLETE.\n"
        "- For place: object on/in target + stable/released = COMPLETE.\n"
        "- For close: door/drawer visibly closed or nearly closed = COMPLETE.\n"
        "- Partial occlusion is acceptable. Use the motion across frames to infer state.\n\n"

        # f"Environment: {env_str}\n"
        # f"Task: {task_str}\n"
        f"Progress history: {history_str}\n"
        f"Previously met criteria: {prior_criteria_str}\n"
        f"Current node type: {node_type}\n"
        f"Current node description: {node_desc}\n"
        f"Current node success criteria: {success_criteria}\n"
    )


StatusPromptBuilder = Callable[
    [str, str, str, list[float] | None, list[str] | None], str
]
STATUS_PROMPT_BUILDERS: dict[str, StatusPromptBuilder] = {
    "standard": status_eval_text,
    "qwen3_5": status_eval_text_qwen3_5,
    "molmo2_8b": status_eval_text_molmo2_8b,
    "gemma4_e4b_it": status_eval_text_gemma4_e4b_it,
}
STATUS_PROMPT_DIR = Path(__file__).resolve().parents[1] / "config" / "status_prompts"
STATUS_PROMPT_FIELDS = {
    "node_type", "node_desc", "success_criteria", "progress_history", "criteria_met_history",
}


@dataclass(frozen=True)
class StatusPrompt:
    """Resolved once per run so edits cannot change a prompt halfway through an episode."""
    name: str
    source: str
    source_text: str
    builder: StatusPromptBuilder
    is_template: bool = False

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()

    def metadata(self) -> dict[str, str]:
        return {"name": self.name, "source": self.source, "sha256": self.sha256}

    def __call__(
        self,
        node_type: str,
        node_desc: str,
        success_criteria: str,
        progress_history: list[float] | None = None,
        criteria_met_history: list[str] | None = None,
    ) -> str:
        return self.builder(node_type, node_desc, success_criteria, progress_history, criteria_met_history)


def load_status_prompt(selector: str = "standard") -> StatusPrompt:
    """Select a legacy builder, config/status_prompts/<id>.txt, or explicit template path."""
    if selector in STATUS_PROMPT_BUILDERS:
        builder = STATUS_PROMPT_BUILDERS[selector]
        return StatusPrompt(
            selector, f"{__file__}:{builder.__name__}", inspect.getsource(builder), builder,
        )
    path = Path(selector).expanduser()
    if not path.is_file():
        path = STATUS_PROMPT_DIR / f"{selector}.txt"
    if not path.is_file():
        raise ValueError(
            f"Unknown status prompt {selector!r}. Use a named builder "
            f"({', '.join(STATUS_PROMPT_BUILDERS)}), an ID under {STATUS_PROMPT_DIR}, "
            "or an existing .txt template path."
        )
    source = path.read_text(encoding="utf-8").rstrip("\n")
    template = Template(source)
    identifiers = {
        match.group("named") or match.group("braced")
        for match in template.pattern.finditer(source)
        if match.group("named") or match.group("braced")
    }
    unknown = identifiers - STATUS_PROMPT_FIELDS
    missing = {"node_type", "node_desc", "success_criteria"} - identifiers
    if unknown or missing:
        raise ValueError(f"Invalid status template {path}: unknown={sorted(unknown)}, missing={sorted(missing)}")
    template.substitute({name: "" for name in STATUS_PROMPT_FIELDS})

    def render(
        node_type: str, node_desc: str, success_criteria: str,
        progress_history: list[float] | None, criteria_met_history: list[str] | None,
    ) -> str:
        history = ", ".join(f"{float(v):.2f}" for v in (progress_history or [])) or "none"
        criteria = ", ".join(str(item) for item in (criteria_met_history or [])) or "none"
        return template.substitute(
            node_type=node_type, node_desc=node_desc, success_criteria=success_criteria,
            progress_history=history, criteria_met_history=criteria,
        )

    return StatusPrompt(path.stem, str(path.resolve()), source, render, is_template=True)