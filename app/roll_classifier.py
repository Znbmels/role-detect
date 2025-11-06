from __future__ import annotations

import json
from typing import Tuple
import requests

ALLOWED_ROLES = {"A-roll", "B-roll", "C-roll"}


SYSTEM_PROMPT = (
    "Classify a single video frame as exactly one of: A-roll, B-roll, C-roll.\n"
    "Definitions:\n"
    "- A-roll: primary narrative track: person facing camera (talking head), narrator, vlog. Indicators: visible face, likely speaking, addressing viewer.\n"
    "- B-roll: supporting visuals without a speaking face: screen/UI/phone, product close-ups, scenery, cutaways, captions on plain background.\n"
    "- C-roll: very short decorative inserts (memes/reactions/micro-cutaways/motion design).\n"
    "If a smartphone UI or interface dominates the frame, prefer B-roll.\n"
    "Respond ONLY with strict JSON: {\"role\": \"A-roll|B-roll|C-roll\", \"confidence\": number 0..1, \"explanation\": \"short reason\"}."
)


def _build_messages(data_uri: str) -> list:
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Classify this frame and include a one-sentence explanation in JSON only."},
                {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
            ],
        },
    ]


def classify_frame(
    azure_endpoint: str,
    api_key: str,
    api_version: str,
    deployment_name: str,
    data_uri: str,
) -> Tuple[str, float, str]:
    """Returns (role, confidence, explanation) using Azure OpenAI REST API."""
    url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    payload = {
        "model": deployment_name,
        "messages": _build_messages(data_uri),
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=None)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content") or "{}"

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        # Try to recover if model added text around JSON
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            payload = json.loads(content[start : end + 1])
        else:
            payload = {"role": "B-roll", "confidence": 0.5, "explanation": ""}

    role = str(payload.get("role", "B-roll"))
    if role not in ALLOWED_ROLES:
        role = "B-roll"

    conf = payload.get("confidence", 0.5)
    try:
        confidence = float(conf)
    except Exception:
        confidence = 0.5

    confidence = max(0.0, min(1.0, confidence))
    explanation = str(payload.get("explanation", "")).strip()
    return role, confidence, explanation


SYSTEM_PROMPT_EXPLAIN = (
    "Given a single frame and its assigned role (A-roll, B-roll, or C-roll), "
    "briefly explain in one short sentence why this role fits, referencing visual cues "
    "like face/speaking, UI/screens, objects, or decorative inserts. Respond ONLY with JSON: {\"explanation\": \"...\"}."
)


def _build_explain_messages(data_uri: str, role: str) -> list:
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT_EXPLAIN},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Role: {role}. Explain briefly in JSON only."},
                {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
            ],
        },
    ]


def explain_frame_reason(
    azure_endpoint: str,
    api_key: str,
    api_version: str,
    deployment_name: str,
    data_uri: str,
    role: str,
) -> str:
    url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    payload = {
        "model": deployment_name,
        "messages": _build_explain_messages(data_uri, role),
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=None)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content") or "{}"
    try:
        payload = json.loads(content)
        explanation = str(payload.get("explanation", ""))
    except Exception:
        explanation = ""
    return explanation
