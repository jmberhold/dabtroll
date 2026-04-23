"""
DABTROLL prompts kept in one place so they're easy to tweak without touching logic.

Keep these prompts close to your notebook wording to avoid breaking behavior.
"""
from __future__ import annotations


def bt_wait_text() -> str:
    """
    Prompt used while waiting for BT synthesis response.
    """
    return "unlock waist: observe environment ready for instruction"


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
        "Failure handling requirement:\n"
        "Use fallback nodes to explicitly encode recovery steps when an action can fail. "
        "Example: [fallback: try action -> recovery action -> condition check]. "
        "Do not end the tree on a single action failure unless the task is unrecoverable.\n\n"
        "Important conventions:\n"
        "- You've merged 'description' and GR00T prompt; use action.description as the instruction text.\n\n"
        "Please infer a reasonable BT with recovery branches from the scene and task, and return ONLY the JSON."
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


def status_eval_text(node_type: str, node_desc: str, success_criteria: str) -> str:
    """
    Prompt for evaluating a single node's status (running/complete/failure) from a short video window.
    """
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
        "- progress_score must be a float in [0.0, 1.0] indicating forward progress toward success criteria where 0.5 represents slightly closer to success than started, and 1.0 represents complete success.\n"
        "- Use temporal evidence across frames; compare early vs late frames.\n"
        "- If progress exists but criteria are not fully met, set status=running and increase progress_score accordingly.\n"
        "- Set status=complete when success criteria are satisfied.\n"
        "- Set status=failure when the goal is unlikely to be achieved right now or a precondition is violated.\n"
        "- Keep notes concise and specific to visible evidence.\n\n"
        "Completion guidance:\n"
        "- Be permissive: if frames show the task goal achieved, mark complete even if hand/arm is not visible.\n"
        "- For placement, accept completion if the object is inside/on target and no longer at source.\n\n"
        f"Node type: {node_type}\n"
        f"Node description: {node_desc}\n"
        f"Success criteria: {success_criteria}"
    )
