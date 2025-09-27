from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import DATASETS
from .schemas import (
    DatasetInfo,
    DatasetListResponse,
    DatasetVideosResponse,
    HealthResponse,
    LocalizationRequest,
    LocalizationResponse,
    LocalizedSegment,
    VideoExamplesResponse,
)
from .service import MS2DTANService, VideoMeta


def _create_service() -> MS2DTANService:
    return MS2DTANService(dataset_configs=DATASETS)


service = _create_service()


def get_service() -> MS2DTANService:
    return service


app = FastAPI(title="MS-2D-TAN Web Demo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health_check(service: MS2DTANService = Depends(get_service)) -> HealthResponse:
    return HealthResponse(device=str(service.device))


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets(service: MS2DTANService = Depends(get_service)) -> DatasetListResponse:
    dataset_info = [DatasetInfo(**meta) for meta in service.get_dataset_metadata()]
    return DatasetListResponse(datasets=dataset_info)


@router.get("/datasets/{dataset_key}/videos", response_model=DatasetVideosResponse)
def list_videos(
    dataset_key: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: MS2DTANService = Depends(get_service),
) -> DatasetVideosResponse:
    try:
        videos, total = service.list_videos(dataset_key, limit=limit, offset=offset)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    meta_available = total > 0
    return DatasetVideosResponse(dataset=dataset_key, videos=videos, total_videos=total, examples_available=meta_available)


@router.get("/datasets/{dataset_key}/videos/{video_id}/examples", response_model=VideoExamplesResponse)
def video_examples(
    dataset_key: str,
    video_id: str,
    service: MS2DTANService = Depends(get_service),
) -> VideoExamplesResponse:
    try:
        meta: VideoMeta = service.get_examples(dataset_key, video_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return VideoExamplesResponse(
        dataset=dataset_key,
        video_id=video_id,
        duration=meta.duration,
        examples=meta.examples,
    )


@router.post("/localize", response_model=LocalizationResponse)
def localize(
    payload: LocalizationRequest,
    service: MS2DTANService = Depends(get_service),
) -> LocalizationResponse:
    try:
        result = service.localize(
            dataset_key=payload.dataset,
            video_id=payload.video_id,
            description=payload.description,
            top_k=payload.top_k,
            nms_threshold=payload.nms_threshold,
        )
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    segments = [LocalizedSegment(**segment) for segment in result['segments']]
    return LocalizationResponse(
        dataset=payload.dataset,
        video_id=payload.video_id,
        description=payload.description,
        duration=result['duration'],
        segments=segments,
        feature_length=result['feature_length'],
        feature_stride=result['feature_stride'],
    )


app.include_router(router)


frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
else:
    @app.get("/")
    def missing_frontend() -> JSONResponse:
        return JSONResponse({"message": "Frontend assets not found. Build the frontend first."})


__all__ = ["app"]
