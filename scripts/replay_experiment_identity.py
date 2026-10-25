"""Content-based identity shared by replay launchers and their resume records."""

import argparse
import hashlib
import json
import math

if __package__:
    from scripts.prompts import load_status_prompt
else:
    from prompts import load_status_prompt


def experiment_identity(
    review_sheet: str, status_prompt: str, model_label: str, output_prefix: str,
    window_source: str, eval_seconds: str = "", window_seconds: str = "",
) -> str:
    timing = [float(value) if value else None for value in (eval_seconds, window_seconds)]
    if any(value is not None and (not math.isfinite(value) or value <= 0) for value in timing):
        raise ValueError("Replay cadence/window seconds must be finite and positive")
    settings = {
        "review_sheet": review_sheet, "prompt_sha256": load_status_prompt(status_prompt).sha256,
        "model_label": model_label, "output_prefix": output_prefix, "window_source": window_source,
        "eval_seconds": timing[0], "window_seconds": timing[1],
    }
    return hashlib.sha256(json.dumps(settings, sort_keys=True, allow_nan=False).encode()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("review_sheet", "status_prompt", "model_label", "output_prefix", "window_source"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--eval-seconds", default="")
    parser.add_argument("--window-seconds", default="")
    print(experiment_identity(**vars(parser.parse_args())))
