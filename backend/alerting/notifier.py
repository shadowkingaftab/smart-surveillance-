"""Async dashboard and mobile-push webhook notifier using the standard library."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import Request, urlopen

from common.models_schema import ThreatAlertPayload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeliveryResult:
    dashboard_subscribers: int
    dashboard_webhook_sent: bool
    mobile_push_sent: bool


class AlertNotifier:
    """Broadcast alerts to in-process dashboard clients and optional HTTP adapters."""

    def __init__(self, *, mobile_push_url: str | None = None,
                 dashboard_webhook_url: str | None = None,
                 bearer_token: str | None = None, timeout_seconds: float = 5.0,
                 dashboard_queue_size: int = 128) -> None:
        if timeout_seconds <= 0 or dashboard_queue_size < 1:
            raise ValueError("timeout_seconds and dashboard_queue_size must be positive")
        for url in (mobile_push_url, dashboard_webhook_url):
            if url and not url.startswith(("https://", "http://")):
                raise ValueError("Notifier endpoints must use HTTP or HTTPS")
        self.mobile_push_url = mobile_push_url
        self.dashboard_webhook_url = dashboard_webhook_url
        self.bearer_token = bearer_token
        self.timeout_seconds = timeout_seconds
        self.dashboard_queue_size = dashboard_queue_size
        self._subscribers: set[asyncio.Queue[dict[str, object]]] = set()

    @classmethod
    def from_config(cls, config: dict[str, object]) -> AlertNotifier:
        return cls(
            mobile_push_url=os.getenv("ALERT_MOBILE_PUSH_URL"),
            dashboard_webhook_url=os.getenv("ALERT_DASHBOARD_WEBHOOK_URL"),
            bearer_token=os.getenv("ALERT_BEARER_TOKEN"),
            timeout_seconds=float(config.get("delivery_timeout_seconds", 5.0)),
            dashboard_queue_size=int(config.get("dashboard_queue_size", 128)),
        )

    async def dispatch(self, alert: ThreatAlertPayload) -> DeliveryResult:
        payload = alert.model_dump(mode="json")
        subscriber_count = self._broadcast(payload)
        calls: list[tuple[str, str]] = []
        if self.mobile_push_url:
            calls.append(("mobile", self.mobile_push_url))
        if self.dashboard_webhook_url:
            calls.append(("dashboard", self.dashboard_webhook_url))
        results = await asyncio.gather(
            *(asyncio.to_thread(self._post_json, url, payload) for _, url in calls),
            return_exceptions=True,
        )
        statuses: dict[str, bool] = {}
        for (channel, _), result in zip(calls, results):
            statuses[channel] = result is True
            if isinstance(result, BaseException):
                logger.error("%s alert delivery failed: %s", channel, result)
        if not subscriber_count and not self.dashboard_webhook_url:
            logger.warning("Alert %s has no dashboard delivery configured", alert.alert_id)
        if not self.mobile_push_url:
            logger.warning("Alert %s has no mobile push adapter configured", alert.alert_id)
        return DeliveryResult(
            dashboard_subscribers=subscriber_count,
            dashboard_webhook_sent=statuses.get("dashboard", False),
            mobile_push_sent=statuses.get("mobile", False),
        )

    async def dashboard_events(self) -> AsyncIterator[dict[str, object]]:
        """Yield live alerts to a dashboard WebSocket endpoint or event consumer."""
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=self.dashboard_queue_size)
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    def _broadcast(self, payload: dict[str, object]) -> int:
        for queue in tuple(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                logger.warning("Dropped dashboard alert because subscriber queue stayed full")
        return len(self._subscribers)

    def _post_json(self, url: str, payload: dict[str, object]) -> bool:
        headers = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return 200 <= response.status < 300
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"POST to configured alert adapter failed: {exc}") from exc
