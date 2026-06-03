"""
DABTROLL prompts kept in one place so they're easy to tweak without touching logic.

Keep these prompts close to your notebook wording to avoid breaking behavior.
"""
from __future__ import annotations


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