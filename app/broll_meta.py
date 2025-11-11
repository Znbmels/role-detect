"""
Модуль для получения метаданных B-roll (время исчезновения talking head).
"""

import base64
import json
import os
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import requests


def get_video_duration(video_path: Path) -> float:
    """Получает длительность видео в секундах."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0.0
    
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    
    if fps and fps > 0:
        return frame_count / fps
    return 0.0


def extract_frame_at_time(video_path: Path, time_seconds: float) -> np.ndarray:
    """Извлекает кадр из видео в указанное время (секунды)."""
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000)
    ret, frame = capture.read()
    capture.release()
    return frame if ret else None


def encode_frame_to_base64(frame: np.ndarray) -> str:
    """Кодирует кадр в base64."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', frame_rgb)
    return base64.b64encode(buffer).decode('utf-8')


def has_talking_head(
    frame: np.ndarray,
    api_endpoint: str,
    api_key: str,
    api_version: str,
    deployment: str,
) -> bool:
    """Проверяет, есть ли talking_head в кадре."""
    frame_base64 = encode_frame_to_base64(frame)
    
    prompt = """Analyze this video frame. Is there a person speaking to the camera (talking head / A-roll) visible?

Respond ONLY with JSON:
{
    "has_talking_head": true/false
}"""
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
                }
            ]
        }
    ]
    
    try:
        url = f"{api_endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }
        payload = {
            "model": deployment,
            "messages": messages,
            "max_tokens": 50,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"].get("content") or "{}"
        result = json.loads(content)
        return result.get("has_talking_head", False)
    except Exception:
        return False


def seconds_to_timestamp(seconds: float) -> str:
    """Преобразует секунды в формат HH:MM:SS."""
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_video_url_from_api(shortcode: str) -> str:
    """Получает video_url из API по shortcode."""
    api_url = f"https://api.recreate.video/getVideo/{shortcode}"
    
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Извлекаем video_url из reel.video_url
        if "reel" in data and "video_url" in data["reel"]:
            video_url = data["reel"]["video_url"]
            return video_url
        else:
            raise ValueError("Поле reel.video_url не найдено в ответе API")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ошибка при запросе к API: {e}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Ошибка при парсинге ответа API: {e}")


def download_video(url: str, output_path: Path) -> Path:
    """Скачивает видео по URL."""
    try:
        urlretrieve(url, output_path)
        return output_path
    except Exception as e:
        raise Exception(f"Ошибка при скачивании видео: {e}")


def find_talking_head_disappear_time(
    video_path: Path,
    api_endpoint: str,
    api_key: str,
    api_version: str,
    deployment: str,
) -> tuple[str, str]:
    """
    Находит точную секунду, когда talking_head исчезает.
    Returns: (time_start, time_end) в формате "HH:MM:SS"
    """
    duration = get_video_duration(video_path)
    
    # Анализируем кадры каждую секунду (FPS=1)
    last_talking_head_time = None
    disappear_time = None
    
    for time_sec in range(int(duration) + 1):
        frame = extract_frame_at_time(video_path, float(time_sec))
        if frame is None:
            continue
        
        has_head = has_talking_head(
            frame, api_endpoint, api_key, api_version, deployment
        )
        
        if has_head:
            last_talking_head_time = time_sec
        elif last_talking_head_time is not None and disappear_time is None:
            # Talking head исчез
            disappear_time = time_sec
            break
    
    # Если talking_head не исчез до конца видео, используем последнюю секунду
    if disappear_time is None:
        disappear_time = int(duration)
    
    # Добавляем запас в 1 секунду, чтобы избежать попадания человека в кадр в миллисекундах
    disappear_time_with_buffer = disappear_time + 1
    # Не выходим за пределы длительности видео
    if disappear_time_with_buffer > int(duration):
        disappear_time_with_buffer = int(duration)
    
    time_start = seconds_to_timestamp(disappear_time_with_buffer)
    time_end = seconds_to_timestamp(int(duration))
    
    return time_start, time_end


