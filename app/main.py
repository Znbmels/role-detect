from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic_settings import BaseSettings
import requests

from .grouping import average_confidence_by_role, group_frames_to_segments, group_frames_with_indices
from .roll_classifier import classify_frame
from .schemas import AnalyzeRequest, AnalyzeResponse, FrameRole, BrollMetaResponse, BrollVideoResponse
from .utils import (
    guess_video_id_from_path,
    image_file_to_data_uri,
    list_image_files,
    url_to_data_uri,
    filename_from_url,
    fetch_image_from_url,
    slice_image_to_tiles,
    pil_image_to_data_uri,
)
from .talking_head import detect_talking_head
from .broll_meta import (
    get_video_url_from_api,
    download_video,
    find_talking_head_disappear_time,
)


class Settings(BaseSettings):
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: str = "2024-08-01-preview"
    azure_openai_deployment: Optional[str] = None  # default deployment name if not provided in request

    class Config:
        env_prefix = "AZURE_OPENAI_"
        case_sensitive = False


# Load .env file manually (из текущей директории)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# иницализация настроек  
settings = Settings(_env_file=os.path.join(os.path.dirname(__file__), "..", ".env"))



# # Load .env if present
# load_dotenv()
# settings = Settings()

app = FastAPI(title="Roll Analyzer API", version="0.1.0")


@app.post("/analyze-rolls", response_model=AnalyzeResponse)
def analyze_rolls(req: AnalyzeRequest) -> AnalyzeResponse:
    frames_path = (req.frames_path or "").strip() or None
    image_urls = req.image_urls or []
    fps = req.fps if req.fps and req.fps > 0 else 30
    deployment = req.deployment_name or settings.azure_openai_deployment

    if not deployment:
        raise HTTPException(status_code=400, detail="Missing deployment name. Provide in request or AZURE_OPENAI_DEPLOYMENT env var.")

    if not frames_path and not image_urls:
        raise HTTPException(status_code=400, detail="Provide either frames_path or image_urls.")

    if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
        raise HTTPException(status_code=500, detail="Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT.")

    frame_roles: List[FrameRole] = []
    frame_data_uris: List[str] = []  # aligned with frame_roles by index; may contain "" if unavailable
    frame_explanations: List[str] = []  # per-frame explanation from classify_frame

    if frames_path:
        try:
            files = list_image_files(frames_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        if not files:
            raise HTTPException(status_code=400, detail="No .jpg/.jpeg/.png files found in provided path.")
        video_id = guess_video_id_from_path(frames_path)
        for path in files:
            data_uri, _ = image_file_to_data_uri(path)
            try:
                role, confidence, explanation, a_ratio, b_ratio = classify_frame(
                    settings.azure_openai_endpoint,
                    settings.azure_openai_api_key,
                    settings.azure_openai_api_version,
                    deployment,
                    data_uri,
                )
            except Exception:
                role, confidence, explanation, a_ratio, b_ratio = "B-roll", 0.5, "", 0.0, 1.0
            frame_roles.append(
                FrameRole(
                    frame=os.path.basename(path),
                    role=role,
                    confidence=confidence,
                    a_role_ratio=a_ratio,
                    b_role_ratio=b_ratio,
                )
            )
            frame_data_uris.append(data_uri)
            frame_explanations.append(explanation)
    else:
        # URL flow
        if not image_urls:
            raise HTTPException(status_code=400, detail="image_urls is empty.")
        video_id = os.path.splitext(filename_from_url(image_urls[0]))[0] or "url"

        # If spritesheet hint provided and single URL → slice grid
        if req.spritesheet_grid_cols and len(image_urls) == 1:
            try:
                img = fetch_image_from_url(image_urls[0])
                tiles = slice_image_to_tiles(img, req.spritesheet_grid_cols, req.spritesheet_grid_rows)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to slice spritesheet: {e}")
            if req.spritesheet_tile_limit and req.spritesheet_tile_limit > 0:
                tiles = tiles[: req.spritesheet_tile_limit]
            # classify each tile
            for idx, tile in enumerate(tiles):
                data_uri = pil_image_to_data_uri(tile, fmt="JPEG")
                try:
                    role, confidence, explanation, a_ratio, b_ratio = classify_frame(
                        settings.azure_openai_endpoint,
                        settings.azure_openai_api_key,
                        settings.azure_openai_api_version,
                        deployment,
                        data_uri,
                    )
                except Exception:
                    role, confidence, explanation, a_ratio, b_ratio = "B-roll", 0.5, "", 0.0, 1.0
                frame_roles.append(
                    FrameRole(
                        frame=f"frame_{idx:02d}.jpg",
                        role=role,
                        confidence=confidence,
                        a_role_ratio=a_ratio,
                        b_role_ratio=b_ratio,
                    )
                )
                frame_data_uris.append(data_uri)
                frame_explanations.append(explanation)
        else:
            # Treat each URL as an individual frame
            for url in image_urls:
                try:
                    data_uri, _ = url_to_data_uri(url)
                except Exception:
                    role, confidence, explanation, a_ratio, b_ratio = "B-roll", 0.5, "", 0.0, 1.0
                    frame_roles.append(
                        FrameRole(
                            frame=filename_from_url(url),
                            role=role,
                            confidence=confidence,
                            a_role_ratio=a_ratio,
                            b_role_ratio=b_ratio,
                        )
                    )
                    frame_data_uris.append("")
                    frame_explanations.append(explanation)
                    continue
                try:
                    role, confidence, explanation, a_ratio, b_ratio = classify_frame(
                        settings.azure_openai_endpoint,
                        settings.azure_openai_api_key,
                        settings.azure_openai_api_version,
                        deployment,
                        data_uri,
                    )
                except Exception:
                    role, confidence, explanation, a_ratio, b_ratio = "B-roll", 0.5, "", 0.0, 1.0
                frame_roles.append(
                    FrameRole(
                        frame=filename_from_url(url),
                        role=role,
                        confidence=confidence,
                        a_role_ratio=a_ratio,
                        b_role_ratio=b_ratio,
                    )
                )
                frame_data_uris.append(data_uri)
                frame_explanations.append(explanation)

    talking_result = detect_talking_head(frame_roles, frame_explanations)

    # Build segments; optionally add explanations using first frame of each segment
    segments_with_idx = group_frames_with_indices(frame_roles, fps)
    segments_out = []
    explanation_limit = req.max_explanations if req.max_explanations and req.max_explanations > 0 else None
    for seg_idx, (segment, s_idx, e_idx) in enumerate(segments_with_idx):
        explanation = ""
        if req.include_explanations and (explanation_limit is None or seg_idx < explanation_limit):
            first_idx = s_idx
            if 0 <= first_idx < len(frame_explanations):
                explanation = frame_explanations[first_idx] or ""
        segment.explanation = explanation

        segment_frames = frame_roles[s_idx:e_idx]
        if segment_frames:
            avg_a = sum(fr.a_role_ratio for fr in segment_frames) / len(segment_frames)
            avg_b = sum(fr.b_role_ratio for fr in segment_frames) / len(segment_frames)
        else:
            avg_a = avg_b = 0.0
        segment.a_role_ratio = round(avg_a, 3)
        segment.b_role_ratio = round(avg_b, 3)
        segments_out.append(segment)
    conf = average_confidence_by_role(frame_roles)

    return AnalyzeResponse(
        video_id=video_id,
        is_talkinghead=talking_result.is_talkinghead,
        talkinghead_confidence=talking_result.confidence,
        talkinghead_evidence=talking_result.evidence or None,
        roles=segments_out,
        frames=frame_roles if req.include_frame_details else None,
        confidence=conf,
    )


@app.get("/")
def root():
    return {"message": "Roll Analyzer API", "docs": "/docs"}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/getBrollMeta/{id}", response_model=BrollMetaResponse)
def get_broll_meta(id: str) -> BrollMetaResponse:
    """
    Получает метаданные B-roll: время начала (когда talking head исчезает) и время окончания видео.
    """
    deployment = settings.azure_openai_deployment
    
    if not deployment:
        raise HTTPException(
            status_code=400,
            detail="Missing deployment name. Set AZURE_OPENAI_DEPLOYMENT env var."
        )
    
    if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
        raise HTTPException(
            status_code=500,
            detail="Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
        )
    
    # Убираем кавычки и пробелы из id
    shortcode = id.strip().strip('"').strip("'")
    
    # Получаем video_url из API
    try:
        video_url = get_video_url_from_api(shortcode)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get video URL: {str(e)}")
    
    # Скачиваем видео во временный файл
    import tempfile
    temp_dir = Path(tempfile.gettempdir())
    temp_video = temp_dir / f"video_{os.getpid()}_{id}.mp4"
    
    try:
        download_video(video_url, temp_video)
        time_start, time_end = find_talking_head_disappear_time(
            temp_video,
            settings.azure_openai_endpoint,
            settings.azure_openai_api_key,
            settings.azure_openai_api_version,
            deployment,
        )
        
        return BrollMetaResponse(time_start=time_start, time_end=time_end)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")
    finally:
        # Удаляем временный файл
        if temp_video.exists():
            temp_video.unlink()


@app.get("/getBrollVideos/{videoId}", response_model=BrollVideoResponse)
def get_broll_videos(videoId: str, request: Request) -> BrollVideoResponse:
    """
    Получает обрезанное B-roll видео по videoId.
    1. Вызывает /getBrollMeta/{videoId} для получения метаданных (time_start, time_end)
    2. Вызывает внешний API для обрезки видео
    3. Возвращает file_url обрезанного видео
    """
    # Убираем кавычки и пробелы из videoId
    shortcode = videoId.strip().strip('"').strip("'")
    
    # Получаем базовый URL для внутреннего запроса
    base_url = str(request.base_url).rstrip('/')
    
    # Вызываем внутренний эндпоинт /getBrollMeta/{id}
    try:
        meta_url = f"{base_url}/getBrollMeta/{shortcode}"
        meta_response = requests.get(meta_url, timeout=300)  # Увеличенный timeout для обработки видео
        meta_response.raise_for_status()
        meta_data = meta_response.json()
        
        time_start = meta_data["time_start"]
        time_end = meta_data["time_end"]
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get B-roll metadata: {str(e)}"
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse metadata response: {str(e)}"
        )
    
    # Вызываем внешний API для обрезки видео
    cut_video_url = (
        f"https://api.recreate.video/cutVideo"
        f"?url=https://videos.rekreate.ai/video?shortcode={shortcode}"
        f"&time_start={time_start}"
        f"&time_end={time_end}"
    )
    
    try:
        response = requests.get(cut_video_url, timeout=300)  # Увеличенный timeout для обрезки видео
        response.raise_for_status()
        result = response.json()
        
        if "file_url" not in result:
            raise HTTPException(
                status_code=500,
                detail="API response does not contain file_url"
            )
        
        return BrollVideoResponse(file_url=result["file_url"])
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to call cutVideo API: {str(e)}"
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse API response: {str(e)}"
        )
