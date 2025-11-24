from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional

from .api import MessageClient
from .types import MessageEnvelope

logger = logging.getLogger("mofox_bus.router")


@dataclass
class TargetConfig:
    url: str
    token: str | None = None
    ssl_verify: str | None = None

    def to_dict(self) -> Dict[str, str | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, str | None]) -> "TargetConfig":
        return cls(
            url=data.get("url", ""),
            token=data.get("token"),
            ssl_verify=data.get("ssl_verify"),
        )


@dataclass
class RouteConfig:
    route_config: Dict[str, TargetConfig]

    def to_dict(self) -> Dict[str, Dict[str, str | None]]:
        return {"route_config": {k: v.to_dict() for k, v in self.route_config.items()}}

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, str | None]]) -> "RouteConfig":
        cfg = {
            platform: TargetConfig.from_dict(target)
            for platform, target in data.get("route_config", {}).items()
        }
        return cls(route_config=cfg)


class Router:
    def __init__(self, config: RouteConfig, custom_logger: logging.Logger | None = None) -> None:
        if custom_logger:
            logger.handlers = custom_logger.handlers
        self.config = config
        self.clients: Dict[str, MessageClient] = {}
        self.handlers: list[Callable[[Dict], None]] = []
        self._running = False
        self._client_tasks: Dict[str, asyncio.Task] = {}
        self._stop_event: asyncio.Event | None = None

    async def connect(self, platform: str) -> None:
        if platform not in self.config.route_config:
            raise ValueError(f"Unknown platform {platform}")
        target = self.config.route_config[platform]
        mode = "tcp" if target.url.startswith(("tcp://", "tcps://")) else "ws"
        if mode != "ws":
            raise NotImplementedError("TCP mode is not implemented yet")
        client = MessageClient(mode="ws")
        client.set_disconnect_callback(self._handle_client_disconnect)
        await client.connect(
            url=target.url,
            platform=platform,
            token=target.token,
            ssl_verify=target.ssl_verify,
        )
        for handler in self.handlers:
            client.register_message_handler(handler)
        self.clients[platform] = client
        if self._running:
            self._start_client_task(platform, client)

    def register_class_handler(self, handler: Callable[[Dict], None]) -> None:
        self.handlers.append(handler)
        for client in self.clients.values():
            client.register_message_handler(handler)

    async def run(self) -> None:
        self._running = True
        self._stop_event = asyncio.Event()
        for platform in self.config.route_config:
            if platform not in self.clients:
                await self.connect(platform)
        for platform, client in self.clients.items():
            if platform not in self._client_tasks:
                self._start_client_task(platform, client)
        try:
            await self._stop_event.wait()
        except asyncio.CancelledError:  # pragma: no cover
            raise

    async def remove_platform(self, platform: str) -> None:
        if platform in self._client_tasks:
            task = self._client_tasks.pop(platform)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        client = self.clients.pop(platform, None)
        if client:
            await client.stop()

    async def _handle_client_disconnect(self, platform: str, reason: str) -> None:
        logger.info("Client for %s disconnected: %s (auto-reconnect handled by client)", platform, reason)
        task = self._client_tasks.get(platform)
        if task is not None and not task.done():
            return
        client = self.clients.get(platform)
        if client and self._running:
            self._start_client_task(platform, client)

    async def stop(self) -> None:
        self._running = False
        if self._stop_event:
            self._stop_event.set()
        for platform in list(self.clients.keys()):
            await self.remove_platform(platform)
        self.clients.clear()

    def _start_client_task(self, platform: str, client: MessageClient) -> None:
        task = asyncio.create_task(client.run())
        task.add_done_callback(lambda t, plat=platform: asyncio.create_task(self._restart_if_needed(plat, t)))
        self._client_tasks[platform] = task

    async def _restart_if_needed(self, platform: str, task: asyncio.Task) -> None:
        if not self._running:
            return
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.warning("Client task for %s ended with exception: %s", platform, exc)
        client = self.clients.get(platform)
        if client:
            self._start_client_task(platform, client)

    def get_target_url(self, message: MessageEnvelope) -> Optional[str]:
        platform = message.get("message_info", {}).get("platform")
        if not platform:
            return None
        target = self.config.route_config.get(platform)
        return target.url if target else None

    async def send_message(self, message: MessageEnvelope):
        platform = message.get("message_info", {}).get("platform")
        if not platform:
            raise ValueError("message_info.platform is required")
        client = self.clients.get(platform)
        if client is None:
            raise RuntimeError(f"No client connected for platform {platform}")
        return await client.send_message(message)

    async def update_config(self, config_data: Dict[str, Dict[str, str | None]]) -> None:
        new_config = RouteConfig.from_dict(config_data)
        await self._adjust_connections(new_config)
        self.config = new_config

    async def _adjust_connections(self, new_config: RouteConfig) -> None:
        current = set(self.config.route_config.keys())
        updated = set(new_config.route_config.keys())
        for platform in current - updated:
            await self.remove_platform(platform)
        for platform in updated:
            if platform not in current:
                await self.connect(platform)
            else:
                old = self.config.route_config[platform]
                new = new_config.route_config[platform]
                if old.url != new.url or old.token != new.token:
                    await self.remove_platform(platform)
                    await self.connect(platform)
