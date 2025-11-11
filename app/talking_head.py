from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .schemas import FrameRole


@dataclass
class TalkingHeadResult:
    is_talkinghead: bool
    confidence: float
    evidence: List[dict[str, str]]


def _max_consecutive(indices: Sequence[int]) -> int:
    if not indices:
        return 0
    longest = 1
    streak = 1
    for prev, curr in zip(indices, indices[1:]):
        if curr == prev + 1:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 1
    return longest


def detect_talking_head(
    frames: List[FrameRole],
    explanations: List[str],
    min_consecutive: int = 3,
) -> TalkingHeadResult:
    """
    Heuristic detection that the overall video is a talking-head composition.
    We look for multiple confident A-roll frames where the on-camera subject
    occupies a significant portion of the frame.
    """
    if not frames:
        return TalkingHeadResult(False, 0.0, [])

    qualified_indices: List[int] = []
    for idx, frame in enumerate(frames):
        if frame.role != "A-roll":
            continue
        if frame.confidence < 0.55:
            continue
        if frame.a_role_ratio < 0.35:
            continue
        qualified_indices.append(idx)

    coverage = len(qualified_indices) / max(1, len(frames))
    longest_run = _max_consecutive(qualified_indices)
    is_talking = longest_run >= min_consecutive or (len(qualified_indices) >= min_consecutive and coverage >= 0.2)

    if is_talking:
        confidence = min(0.99, round(0.6 * coverage + 0.4 * min(1.0, longest_run / (min_consecutive + 1)), 3))
    else:
        confidence = round(coverage * 0.5, 3)

    evidence: List[dict[str, str]] = []
    for idx in qualified_indices[:3]:
        description = explanations[idx] if idx < len(explanations) and explanations[idx] else "Face-forward frame."
        evidence.append({"frame": frames[idx].frame, "description": description})

    return TalkingHeadResult(is_talking, confidence, evidence)
