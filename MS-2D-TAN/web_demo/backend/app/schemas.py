from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, constr, conint, confloat


DatasetKey = constr(strip_whitespace=True, to_lower=True, min_length=1)
VideoId = constr(strip_whitespace=True, min_length=1)


class DatasetInfo(BaseModel):
    key: str
    display_name: str
    description: str
    cfg_path: str
    checkpoint_path: str
    feature_requirements: str
    best_metrics: dict
    default_top_k: int
    notes: List[str]


class DatasetListResponse(BaseModel):
    datasets: List[DatasetInfo]


class DatasetVideosResponse(BaseModel):
    dataset: str
    videos: List[str]
    total_videos: int
    examples_available: bool = False


class VideoExamplesResponse(BaseModel):
    dataset: str
    video_id: str
    duration: float
    examples: List[dict]


class LocalizationRequest(BaseModel):
    dataset: DatasetKey = Field(..., description="Dataset key as defined by the backend, e.g., 'charades'.")
    video_id: VideoId = Field(..., description="Video identifier matching the precomputed feature file.")
    description: str = Field(..., min_length=3, description="Natural language query to localize.")
    top_k: conint(strict=True, ge=1, le=10) = Field(5, description="How many proposals to return.")
    nms_threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(None, description="Override the default NMS IoU threshold.")


class LocalizedSegment(BaseModel):
    start: float
    end: float
    score: float


class LocalizationResponse(BaseModel):
    dataset: str
    video_id: str
    description: str
    duration: float
    segments: List[LocalizedSegment]
    feature_length: int
    feature_stride: float


class HealthResponse(BaseModel):
    status: str = "ok"
    device: str
