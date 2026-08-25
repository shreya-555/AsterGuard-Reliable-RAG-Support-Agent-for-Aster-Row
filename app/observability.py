import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SENSITIVE_KEYS = {
    "customer",
    "name",
    "email",
    "address",
    "shipping_address",
    "internal",
    "internal_notes",
    "warehouse_note",
    "risk_score",
    "support_tags",
    "api_key",
    "authorization",
}

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
API_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
GIFT_CODE_RE = re.compile(
    r"(?i)(gift[- ]?card\s+code\s*[:=]?\s*)([A-Za-z0-9-]{6,})"
)


class TraceLogger:
    """Tiny structured JSONL trace logger with recursive redaction."""

    def __init__(
        self,
        enabled: bool = False,
        path: str | Path = "logs/agent.jsonl",
        echo: bool = False,
    ):
        self.enabled = enabled
        self.path = Path(path)
        self.echo = echo

    def log(self, event: str, **payload: Any) -> None:
        if not self.enabled:
            return

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **self._sanitize(payload),
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)

        with self.path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

        if self.echo:
            print(f"[debug] {line}")

    @classmethod
    def _sanitize(cls, value: Any):
        if isinstance(value, dict):
            cleaned = {}
            for key, item in value.items():
                if str(key).lower() in SENSITIVE_KEYS:
                    cleaned[key] = "[REDACTED]"
                else:
                    cleaned[key] = cls._sanitize(item)
            return cleaned

        if isinstance(value, list):
            return [cls._sanitize(item) for item in value]

        if isinstance(value, tuple):
            return [cls._sanitize(item) for item in value]

        if isinstance(value, str):
            value = EMAIL_RE.sub("[REDACTED_EMAIL]", value)
            value = API_KEY_RE.sub("[REDACTED_API_KEY]", value)
            value = GIFT_CODE_RE.sub(r"\1[REDACTED_GIFT_CODE]", value)
            return value

        return value
