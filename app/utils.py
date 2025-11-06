import base64
import os
from typing import List, Tuple, Optional
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from io import BytesIO
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_image_files(frames_path: str) -> List[str]:
    if not os.path.isdir(frames_path):
        raise FileNotFoundError(f"Frames path does not exist: {frames_path}")
    files = [
        os.path.join(frames_path, f)
        for f in os.listdir(frames_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]
    files.sort()
    return files


def guess_video_id_from_path(frames_path: str) -> str:
    base = os.path.basename(os.path.normpath(frames_path))
    return base or "video"


def seconds_to_timecode(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total_seconds = int(round(seconds))
    mm = total_seconds // 60
    ss = total_seconds % 60
    return f"{mm:02d}:{ss:02d}"


def frames_to_seconds(frame_index: int, fps: int) -> float:
    if fps and fps > 0:
        return frame_index / float(fps)
    return float(frame_index)


def image_file_to_data_uri(path: str) -> Tuple[str, str]:
    _, ext = os.path.splitext(path)
    ext_l = ext.lower().lstrip(".")
    if ext_l == "jpg":
        ext_l = "jpeg"
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/{ext_l};base64,{b64}", ext_l


def filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = os.path.basename(parsed.path)
    return name or "image.jpg"


def url_to_data_uri(url: str) -> Tuple[str, str]:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as resp:
        content_type = resp.headers.get("Content-Type", "").lower()
        data = resp.read()
    if "jpeg" in content_type or "jpg" in content_type:
        ext_l = "jpeg"
    elif "png" in content_type:
        ext_l = "png"
    else:
        # Guess from URL path
        path_ext = os.path.splitext(urlparse(url).path)[1].lower().lstrip(".")
        if path_ext in {"jpg", "jpeg"}:
            ext_l = "jpeg"
        elif path_ext == "png":
            ext_l = "png"
        else:
            ext_l = "jpeg"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/{ext_l};base64,{b64}", ext_l


def fetch_image_from_url(url: str) -> Image.Image:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as resp:
        data = resp.read()
    return Image.open(BytesIO(data)).convert("RGB")


def slice_image_to_tiles(img: Image.Image, grid_cols: int, grid_rows: Optional[int] = None) -> List[Image.Image]:
    if grid_cols <= 0:
        raise ValueError("grid_cols must be > 0")
    width, height = img.size
    if grid_rows is None or grid_rows <= 0:
        # infer rows by assuming square-ish tiles
        tile_w = width // grid_cols
        grid_rows = max(1, height // max(1, tile_w))
    tile_w = width // grid_cols
    tile_h = height // grid_rows
    tiles: List[Image.Image] = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            left = c * tile_w
            upper = r * tile_h
            right = left + tile_w
            lower = upper + tile_h
            tiles.append(img.crop((left, upper, right, lower)))
    return tiles


def pil_image_to_data_uri(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt, quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    ext_l = "jpeg" if fmt.upper() == "JPEG" else fmt.lower()
    return f"data:image/{ext_l};base64,{b64}"
