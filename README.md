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
  "is_talkinghead": true,
  "talkinghead_confidence": 0.87,
  "talkinghead_evidence": [
    {
      "frame": "frame_0003.jpg",
      "description": "Host looks into camera, medium close-up."
    }
  ],
  "roles": [
    { "start": "00:00", "end": "00:02", "role": "C-roll", "a_role_ratio": 0.0, "b_role_ratio": 1.0 },
    { "start": "00:02", "end": "00:10", "role": "A-roll", "a_role_ratio": 0.52, "b_role_ratio": 0.48 }
  ],
  "frames": [
    { "frame": "frame_0001.jpg", "role": "A-roll", "confidence": 0.95, "a_role_ratio": 0.9, "b_role_ratio": 0.1 }
  ],
  "confidence": { "A-roll": 0.94, "B-roll": 0.89 }
}
```

- `a_role_ratio` captures how much of the frame/segment is occupied by the speaking person (A-roll) versus supporting visuals (`b_role_ratio`). The values stay between 0 and 1 and roughly sum to 1.
- Segment explanations reuse the first frame of each segment; cap how many are generated via `max_explanations`.

### Talking head detection

- Before interpreting the roll segments, check `is_talkinghead`. The detector looks for several confident A-roll frames where the on-camera speaker occupies at least ~35% of the frame and appears in consecutive frames.
- `talkinghead_confidence` is a heuristic score (0-1) derived from proportion of such frames and their longest consecutive run.
- `talkinghead_evidence` lists a few representative frames with short explanations so you can inspect why the clip was judged as a talking head.
- If `is_talkinghead = false` you can skip downstream A/B analysis entirely (segments/frames are still returned for debugging or alternative uses).

Notes:

- Frames are processed sequentially. Each image is sent to the Azure GPT-4o deployment with a strict JSON response instruction.
- Timecodes are computed from FPS; end time of a segment equals the start time of the next segment.
- If `include_frame_details` is false, the `frames` field is omitted.

## Tips

- Name frames in natural sort order (e.g., `frame_0001.jpg`, `frame_0002.jpg`, ...).
- To speed up or reduce cost, consider sampling frames (e.g., every Nth frame) or batching (future enhancement).
