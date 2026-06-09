from __future__ import annotations

import json
import logging
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)


class WebhookAlert:
    def __init__(self, url: str | None, timeout_seconds: float = 3.0) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds

    def send_unknown(self, confidence: float, source: str) -> bool:
        if not self.url:
            return False
        payload = json.dumps(
            {
                "event": "unknown_person",
                "confidence": confidence,
                "source": source,
            }
        ).encode("utf-8")
        request = Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return 200 <= response.status < 300
        except OSError:
            LOGGER.exception("Unknown-person webhook delivery failed.")
            return False
