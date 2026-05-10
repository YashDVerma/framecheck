import json
import os
import re

import requests

# Cached on Anthropic's side — the instructions never change, only the user text does.
_SYSTEM_PROMPT = """\
You analyze news text for journalistic framing. When given a news passage, return ONLY a JSON object with exactly these keys:
- "frame": a 5-10 word label for the dominant narrative frame (e.g. "conflict frame", "accountability journalism frame")
- "loaded_language": a list of words or phrases that carry strong positive or negative connotation
- "rationale": one sentence explaining your framing classification

Return raw JSON only — no markdown fences, no commentary."""


def frame_summary(text: str) -> dict | None:
    backend = os.getenv("LLM_BACKEND", "ollama")
    if backend == "anthropic":
        return _anthropic_call(text)
    if backend == "none":
        return None
    return _ollama_call(text)


def _parse_json(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    return json.loads(cleaned)


def _ollama_call(text: str) -> dict | None:
    prompt = f"{_SYSTEM_PROMPT}\n\nText: {text}\n\nJSON:"
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
                "prompt": prompt,
                "stream": False,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return _parse_json(resp.json()["response"])
    except (requests.ConnectionError, requests.Timeout):
        return None  # Ollama not running — degrade gracefully rather than crash


def _anthropic_call(text: str) -> dict:
    import anthropic  # lazy — only needed when LLM_BACKEND=anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # prompt caching — system prompt is reused across calls
            }
        ],
        messages=[{"role": "user", "content": text}],
    )
    return _parse_json(msg.content[0].text)
