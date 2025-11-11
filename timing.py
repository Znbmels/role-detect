#!/usr/bin/env python3
"""
Находит точную секунду, когда talking_head исчезает из видео.
"""

import base64
import json
import os
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import requests
from openai import OpenAI

# Импортируем настройки из config.py
sys.path.insert(0, str(Path(__file__).parent / "edit"))
try:
    from config import (
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_DEPLOYMENT,
        AZURE_OPENAI_API_VERSION
    )
    api_key = AZURE_OPENAI_API_KEY
    api_base = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}"
    model = AZURE_OPENAI_DEPLOYMENT
    api_version = AZURE_OPENAI_API_VERSION
except ImportError:
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    api_base = None
    model = "gpt-4o"
    api_version = None


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


def has_talking_head(frame: np.ndarray, client: OpenAI, time_sec: float) -> bool:
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
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=50,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("has_talking_head", False)
    except:
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


def find_talking_head_disappear_time(video_path: Path) -> tuple:
    """
    Находит точную секунду, когда talking_head исчезает.
    Returns: (time_start, time_end) в формате "00:00:nn"
    """
    duration = get_video_duration(video_path)
    
    # Создаем клиент OpenAI
    client_kwargs = {"api_key": api_key}
    if api_base:
        client_kwargs["base_url"] = api_base
        if api_version:
            client_kwargs["default_query"] = {"api-version": api_version}
    client = OpenAI(**client_kwargs)
    
    # Анализируем кадры каждую секунду (FPS=1)
    last_talking_head_time = None
    disappear_time = None
    
    for time_sec in range(int(duration) + 1):
        frame = extract_frame_at_time(video_path, float(time_sec))
        if frame is None:
            continue
        
        has_head = has_talking_head(frame, client, time_sec)
        
        if has_head:
            last_talking_head_time = time_sec
        elif last_talking_head_time is not None and disappear_time is None:
            # Talking head исчез
            disappear_time = time_sec
            break
    
    # Если talking_head не исчез до конца видео, используем последнюю секунду
    if disappear_time is None:
        disappear_time = int(duration)
    
    time_start = seconds_to_timestamp(disappear_time)
    time_end = seconds_to_timestamp(int(duration))
    
    return time_start, time_end


if __name__ == "__main__":
    # Получаем shortcode из аргументов командной строки или из input
    if len(sys.argv) > 1:
        shortcode = sys.argv[1].strip()
    else:
        shortcode = input("Введите shortcode: ").strip()
    
    if not shortcode:
        print("❌ Shortcode не указан", flush=True)
        sys.exit(1)
    
    # Убираем кавычки и пробелы
    shortcode = shortcode.strip().strip('"').strip("'")
    
    # Получаем video_url из API
    try:
        video_url = get_video_url_from_api(shortcode)
    except Exception as e:
        print(f"❌ Ошибка: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    
    # Скачиваем видео во временный файл
    temp_dir = Path(tempfile.gettempdir())
    temp_video = temp_dir / f"video_{os.getpid()}.mp4"
    
    try:
        download_video(video_url, temp_video)
        time_start, time_end = find_talking_head_disappear_time(temp_video)
        
        print(f"time_start={time_start}")
        print(f"time_end={time_end}")
    except Exception as e:
        print(f"❌ Ошибка: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    finally:
        # Удаляем временный файл
        if temp_video.exists():
            temp_video.unlink()
