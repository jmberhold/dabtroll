"""Video-clock helpers shared by replay and offline analysis."""

from pathlib import Path
from typing import Optional


def _fmt_mmss_from_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    total_seconds = max(0.0, float(seconds))
    minutes = int(total_seconds // 60)
    rem_seconds = int(round(total_seconds - (minutes * 60)))
    if rem_seconds == 60:
        minutes += 1
        rem_seconds = 0
    return f"{minutes}:{rem_seconds:02d}"


def _safe_video_duration_seconds(video_path: Path) -> Optional[float]:
    if not video_path.exists():
        return None

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()
            if fps > 0.0 and frame_count >= 0.0:
                return frame_count / fps
        else:
            cap.release()
    except Exception:
        pass

    try:
        import av  # type: ignore

        with av.open(str(video_path)) as container:
            if container.duration is not None:
                return float(container.duration) / 1_000_000.0
    except Exception:
        pass

    return None


def _find_primary_video_path(episode_dir: Path) -> Optional[Path]:
    video_dir = episode_dir / "video"
    if not video_dir.exists():
        return None
    mp4s = sorted(video_dir.glob("**/*.mp4"))
    return mp4s[0] if mp4s else None
