from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


RoleType = Literal["A-roll", "B-roll", "C-roll"]


class AnalyzeRequest(BaseModel):
    frames_path: Optional[str] = Field(None, description="Absolute path to directory with frames (.jpg/.png)")
    image_urls: Optional[List[str]] = Field(None, description="Optional list of image URLs to analyze")
    spritesheet_grid_cols: Optional[int] = Field(None, description="If provided with a single image URL, slice it into this many columns")
    spritesheet_grid_rows: Optional[int] = Field(None, description="Optional number of rows for spritesheet slicing (if unknown, will infer)")
    spritesheet_tile_limit: Optional[int] = Field(None, description="Optional max number of tiles to use from the spritesheet (row-major order)")
    fps: Optional[int] = Field(None, description="Frames per second for timecode grouping")
    deployment_name: Optional[str] = Field(None, description="Azure OpenAI deployment name (e.g., gpt-4o)")
    include_frame_details: bool = Field(True, description="Include per-frame role classifications in response")
    include_explanations: bool = Field(False, description="If true, generate brief explanation per segment (adds extra model calls)")
    max_explanations: Optional[int] = Field(None, description="Optional cap on number of segments to explain (first N)")


class TalkingHeadEvidence(BaseModel):
    frame: str
    description: str


class FrameRole(BaseModel):
    frame: str
    role: RoleType
    confidence: float
    a_role_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Portion of the frame occupied by the on-camera speaker.")
    b_role_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Portion of the frame occupied by supporting visuals.")


class Segment(BaseModel):
    start: str
    end: str
    role: RoleType
    explanation: Optional[str] = None
    a_role_ratio: float = Field(0.0, ge=0.0, le=1.0)
    b_role_ratio: float = Field(0.0, ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    video_id: str
    is_talkinghead: bool
    talkinghead_confidence: float = Field(0.0, ge=0.0, le=1.0)
    talkinghead_evidence: Optional[List[TalkingHeadEvidence]] = None
    roles: List[Segment]
    frames: Optional[List[FrameRole]]
    confidence: Dict[RoleType, float]


class BrollMetaResponse(BaseModel):
    time_start: str = Field(..., description="Time when talking head disappears (format: HH:MM:SS)")
    time_end: str = Field(..., description="End time of video (format: HH:MM:SS)")


class BrollVideoResponse(BaseModel):
    file_url: str = Field(..., description="URL of the cut video file")
