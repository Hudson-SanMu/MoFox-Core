"""发送处理器 - 将 MessageEnvelope 转换并发送到 Napcat"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from src.common.logger import get_logger

if TYPE_CHECKING:
    from ...plugin import NapcatAdapter

logger = get_logger("napcat_adapter.send_handler")


class SendHandler:
    """处理向 Napcat 发送消息"""

    def __init__(self, adapter: "NapcatAdapter"):
        self.adapter = adapter
        self.plugin_config: Optional[Dict[str, Any]] = None

    def set_plugin_config(self, config: Dict[str, Any]) -> None:
        """设置插件配置"""
        self.plugin_config = config

    async def handle_message(self, envelope) -> None:
        """
        处理发送消息
        
        将 MessageEnvelope 转换为 OneBot API 调用
        """
        message_info = envelope.get("message_info", {})
        message_segment = envelope.get("message_segment", {})
        
        # 获取群组和用户信息
        group_info = message_info.get("group_info")
        user_info = message_info.get("user_info")
        
        # 构造消息内容
        message = self._convert_seg_to_onebot(message_segment)
        
        # 发送消息
        if group_info:
            # 发送群消息
            group_id = group_info.get("group_id")
            if group_id:
                await self.adapter.send_napcat_api("send_group_msg", {
                    "group_id": int(group_id),
                    "message": message,
                })
        elif user_info:
            # 发送私聊消息
            user_id = user_info.get("user_id")
            if user_id:
                await self.adapter.send_napcat_api("send_private_msg", {
                    "user_id": int(user_id),
                    "message": message,
                })

    def _convert_seg_to_onebot(self, seg: Dict[str, Any]) -> list:
        """将 SegPayload 转换为 OneBot 消息格式"""
        seg_type = seg.get("type", "")
        seg_data = seg.get("data", "")
        
        if seg_type == "text":
            return [{"type": "text", "data": {"text": seg_data}}]
        elif seg_type == "image":
            return [{"type": "image", "data": {"file": f"base64://{seg_data}"}}]
        elif seg_type == "seglist":
            # 递归处理列表
            result = []
            for sub_seg in seg_data:
                result.extend(self._convert_seg_to_onebot(sub_seg))
            return result
        else:
            # 默认作为文本
            return [{"type": "text", "data": {"text": str(seg_data)}}]
