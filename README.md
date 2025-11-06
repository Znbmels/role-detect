# Roll Analyzer API (A-roll / B-roll / C-roll)

FastAPI service that classifies video frames (jpg/png) into A-roll, B-roll, or C-roll using Azure OpenAI GPT-4o. It then groups contiguous frames into time segments based on FPS.

## Setup

1. Python 3.10+
2. Create and activate a virtual environment
3. Install dependencies
4. Configure Azure OpenAI credentials

```bash
cd /Users/zainab/Desktop/roles
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with your values
```

Required env vars:
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT` (e.g., `https://YOUR-RESOURCE.openai.azure.com`)
- `AZURE_OPENAI_API_VERSION` (default `2024-08-01-preview`)
- `AZURE_OPENAI_DEPLOYMENT` (e.g., `gpt-4o`)

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API

### POST /analyze-rolls

Request body:

```json
{
  "frames_path": "/absolute/path/to/frames",
  "fps": 30,
  "deployment_name": "gpt-4o",
  "include_frame_details": true
}
```

Response body (example):

```json
{
  "video_id": "clip_001",
  "roles": [
    {"start": "00:00", "end": "00:24", "role": "A-roll"},
    {"start": "00:25", "end": "00:35", "role": "B-roll"}
  ],
  "frames": [
    {"frame": "frame_0001.jpg", "role": "A-roll", "confidence": 0.95},
    {"frame": "frame_0002.jpg", "role": "A-roll", "confidence": 0.92}
  ],
  "confidence": {"A-roll": 0.94, "B-roll": 0.89}
}
```

Notes:
- Frames are processed sequentially. Each image is sent to the Azure GPT-4o deployment with a strict JSON response instruction.
- Timecodes are computed from FPS; end time of a segment equals the start time of the next segment.
- If `include_frame_details` is false, the `frames` field is omitted.

## Tips
- Name frames in natural sort order (e.g., `frame_0001.jpg`, `frame_0002.jpg`, ...).
- To speed up or reduce cost, consider sampling frames (e.g., every Nth frame) or batching (future enhancement).
