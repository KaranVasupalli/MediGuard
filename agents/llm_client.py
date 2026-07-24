"""LLM client: local Ollama first, optional cloud fallback, offline template last.

Three providers, chosen by config — the code never changes, only the setting:
  ollama              - local model on your machine (default; private, no API key)
  openai_compatible   - any hosted API speaking the OpenAI chat format
  offline             - deterministic template, no model at all

The offline mode is not a toy. It means the whole pipeline runs and is testable
without a model present (CI, this sandbox, a laptop with Ollama stopped), and it
guarantees a verdict is always produced. Its output is plainly marked as templated so
nobody mistakes it for model-written text.

Responses are cached on disk by a hash of (provider, model, prompt). Temperature is 0
everywhere: the same claim must produce the same explanation, or an audit is
meaningless.
"""
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from config.spark_config import load_config


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, cfg: dict | None = None):
        full = cfg or load_config()
        self.cfg = full.get("llm", {})
        self.provider = self.cfg.get("provider", "ollama")
        self.cache_dir = Path(self.cfg.get("cache_dir", "./data/_llm_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_source = None          # which provider actually answered

    # ---------- cache ----------
    def _cache_key(self, model: str, prompt: str, system: str) -> str:
        h = hashlib.sha256(f"{self.provider}|{model}|{system}|{prompt}".encode())
        return h.hexdigest()[:32]

    def _cached(self, key: str) -> str | None:
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            try:
                return json.loads(f.read_text())["response"]
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def _store(self, key: str, prompt: str, response: str):
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps({"prompt": prompt, "response": response}, indent=2))

    # ---------- providers ----------
    def _call_ollama(self, prompt: str, system: str) -> str:
        oc = self.cfg.get("ollama", {})
        url = oc.get("base_url", "http://localhost:11434").rstrip("/") + "/api/generate"
        body = json.dumps({
            "model": oc.get("model", "qwen2.5:3b"),
            "prompt": prompt, "system": system, "stream": False,
            "options": {"temperature": 0},
        }).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["response"].strip()
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as e:
            raise LLMError(f"ollama unavailable: {type(e).__name__}") from e

    def _call_openai_compatible(self, prompt: str, system: str) -> str:
        fb = self.cfg.get("fallback", {})
        key = os.environ.get(fb.get("api_key_env", "LLM_API_KEY"), "")
        base = (fb.get("base_url") or "").rstrip("/")
        if not key or not base:
            raise LLMError("cloud fallback not configured (missing base_url or API key env)")
        body = json.dumps({
            "model": fb.get("model", ""), "temperature": 0,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            base + "/chat/completions", data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, OSError, KeyError, IndexError,
                json.JSONDecodeError) as e:
            raise LLMError(f"cloud fallback failed: {type(e).__name__}") from e

    # ---------- public ----------
    def generate(self, prompt: str, system: str = "", allow_offline: bool = True) -> str:
        """Try the configured provider, then cloud fallback, then offline template."""
        model = self.cfg.get("ollama", {}).get("model", "offline")
        key = self._cache_key(model, prompt, system)
        hit = self._cached(key)
        if hit is not None:
            self.last_source = "cache"
            return hit

        errors = []
        order = []
        if self.provider == "ollama":
            order.append(("ollama", self._call_ollama))
            if self.cfg.get("fallback", {}).get("enabled"):
                order.append(("cloud", self._call_openai_compatible))
        elif self.provider == "openai_compatible":
            order.append(("cloud", self._call_openai_compatible))

        for name, fn in order:
            try:
                out = fn(prompt, system)
                self.last_source = name
                self._store(key, prompt, out)
                return out
            except LLMError as e:
                errors.append(str(e))

        if not allow_offline:
            raise LLMError("; ".join(errors) or "no provider available")
        self.last_source = "offline"
        return ""          # caller falls back to its deterministic template

    def describe(self) -> dict:
        """Audit metadata about which model is configured."""
        oc = self.cfg.get("ollama", {})
        return {"llm_provider": self.provider,
                "llm_model": oc.get("model", "n/a"),
                "temperature": 0}
