"""通知事件处理器"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from src.common.logger import get_logger

if TYPE_CHECKING:
    from ...plugin import NapcatAdapter

logger = get_logger("napcat_adapter.notice_handler")


class NoticeHandler:
    """处理 Napcat 通知事件（戳一戳、表情回复等）"""

    def __init__(self, adapter: "NapcatAdapter"):
        self.adapter = adapter
        self.plugin_config: Optional[Dict[str, Any]] = None

    def set_plugin_config(self, config: Dict[str, Any]) -> None:
        """设置插件配置"""
        self.plugin_config = config

    async def handle_notice(self, raw: Dict[str, Any]):
        """处理通知事件"""
        # 简化版本：返回一个空的 MessageEnvelope
        import time
        import uuid
        
        return {
            "direction": "incoming",
            "message_info": {
                "platform": "qq",
                "message_id": str(uuid.uuid4()),
                "time": time.time(),
            },
            "message_segment": {"type": "text", "data": "[通知事件]"},
            "timestamp_ms": int(time.time() * 1000),
        }
