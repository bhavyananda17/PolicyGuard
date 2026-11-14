"""Groq chat client with a disk cache, JSON parsing and free-tier rate-limit handling.

Every request/response is cached under data/cache/, so re-running a script never spends
quota twice and every LLM output used in the dataset can be audited.
"""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data/cache")
URL = "https://api.groq.com/openai/v1/chat/completions"


def _key():
    k = os.environ.get("GROQ_API_KEY")
    if not k:
        with open(os.path.join(ROOT, ".env")) as f:
            for line in f:
                if line.startswith("GROQ_API_KEY="):
                    k = line.split("=", 1)[1].strip().strip('"')
    if not k:
        raise RuntimeError("GROQ_API_KEY missing from environment and .env")
    return k


USAGE = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0}


def chat(messages, model="openai/gpt-oss-120b", max_tokens=1500, temperature=0.3,
         json_mode=False, reasoning_effort="low", tag=""):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = reasoning_effort
    elif model.startswith("qwen/"):
        body["reasoning_effort"] = "none"               # no hidden thinking: saves the daily token budget
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]
    path = os.path.join(CACHE, f"{tag + '_' if tag else ''}{h}.json")
    if os.path.exists(path):
        USAGE["cached"] += 1
        with open(path) as f:
            return json.load(f)["content"]

    data = json.dumps(body).encode()
    for attempt in range(8):
        req = urllib.request.Request(URL, data=data, headers={
            "Authorization": "Bearer " + _key(), "Content-Type": "application/json", "User-Agent": "policyguard"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code == 429 or e.code >= 500:
                wait = float(e.headers.get("retry-after") or 0) or min(60, 4 * 2 ** attempt)
                if "per day" in msg or "TPD" in msg or "RPD" in msg:
                    raise RuntimeError(f"daily limit hit on {model}: {msg}")
                time.sleep(wait + 1)
                continue
            if e.code == 400 and json_mode and "json" in msg.lower():
                body.pop("response_format", None)        # model refused strict JSON: parse leniently
                data = json.dumps(body).encode()
                continue
            raise RuntimeError(f"Groq {e.code}: {msg}")
    else:
        raise RuntimeError("Groq: retries exhausted")

    content = out["choices"][0]["message"].get("content") or ""
    u = out.get("usage", {})
    USAGE["calls"] += 1
    USAGE["prompt_tokens"] += u.get("prompt_tokens", 0)
    USAGE["completion_tokens"] += u.get("completion_tokens", 0)
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"request": body, "content": content, "usage": u}, f)
    return content


def parse_json(text):
    """First JSON object in text, or None."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None
