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


class FrameRole(BaseModel):
    frame: str
    role: RoleType
    confidence: float


class Segment(BaseModel):
    start: str
    end: str
    role: RoleType
    explanation: Optional[str] = None


class AnalyzeResponse(BaseModel):
    video_id: str
    roles: List[Segment]
    frames: Optional[List[FrameRole]]
    confidence: Dict[RoleType, float]
