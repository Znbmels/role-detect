from __future__ import annotations

from typing import Dict, List, Tuple

from .schemas import FrameRole, Segment, RoleType
from .utils import frames_to_seconds, seconds_to_timecode


def group_frames_to_segments(frames: List[FrameRole], fps: int | None) -> List[Segment]:
    if not frames:
        return []

    segments: List[Segment] = []
    start_idx = 0
    current_role: RoleType = frames[0].role

    for idx in range(1, len(frames)):
        if frames[idx].role != current_role:
            start_s = frames_to_seconds(start_idx, fps or 1)
            end_s = frames_to_seconds(idx, fps or 1)
            segments.append(
                Segment(
                    start=seconds_to_timecode(start_s),
                    end=seconds_to_timecode(end_s),
                    role=current_role,
                )
            )
            start_idx = idx
            current_role = frames[idx].role

    # tail
    start_s = frames_to_seconds(start_idx, fps or 1)
    end_s = frames_to_seconds(len(frames), fps or 1)
    segments.append(
        Segment(
            start=seconds_to_timecode(start_s),
            end=seconds_to_timecode(end_s),
            role=current_role,
        )
    )
    return segments


def average_confidence_by_role(frames: List[FrameRole]) -> Dict[RoleType, float]:
    sums: Dict[RoleType, float] = {"A-roll": 0.0, "B-roll": 0.0, "C-roll": 0.0}
    counts: Dict[RoleType, int] = {"A-roll": 0, "B-roll": 0, "C-roll": 0}

    for fr in frames:
        sums[fr.role] += fr.confidence
        counts[fr.role] += 1

    result: Dict[RoleType, float] = {}
    for k in sums:
        if counts[k] > 0:
            result[k] = round(sums[k] / counts[k], 3)
    return result


def group_frames_with_indices(frames: List[FrameRole], fps: int | None) -> List[tuple[Segment, int, int]]:
    """
    Returns list of (segment, start_index, end_index), where end_index is exclusive.
    Mirrors group_frames_to_segments but keeps original frame indices to support explanations.
    """
    if not frames:
        return []

    grouped: List[tuple[Segment, int, int]] = []
    start_idx = 0
    current_role: RoleType = frames[0].role

    for idx in range(1, len(frames)):
        if frames[idx].role != current_role:
            start_s = frames_to_seconds(start_idx, fps or 1)
            end_s = frames_to_seconds(idx, fps or 1)
            grouped.append(
                (
                    Segment(
                        start=seconds_to_timecode(start_s),
                        end=seconds_to_timecode(end_s),
                        role=current_role,
                    ),
                    start_idx,
                    idx,
                )
            )
            start_idx = idx
            current_role = frames[idx].role

    start_s = frames_to_seconds(start_idx, fps or 1)
    end_s = frames_to_seconds(len(frames), fps or 1)
    grouped.append(
        (
            Segment(
                start=seconds_to_timecode(start_s),
                end=seconds_to_timecode(end_s),
                role=current_role,
            ),
            start_idx,
            len(frames),
        )
    )
    return grouped
