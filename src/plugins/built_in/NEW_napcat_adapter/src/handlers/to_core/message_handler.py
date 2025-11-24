"""消息处理器 - 将 Napcat OneBot 消息转换为 MessageEnvelope"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.common.logger import get_logger
from src.plugin_system.apis import config_api

if TYPE_CHECKING:
    from ...plugin import NapcatAdapter

logger = get_logger("napcat_adapter.message_handler")


class MessageHandler:
    """处理来自 Napcat 的消息事件"""

    def __init__(self, adapter: "NapcatAdapter"):
        self.adapter = adapter
        self.plugin_config: Optional[Dict[str, Any]] = None

    def set_plugin_config(self, config: Dict[str, Any]) -> None:
        """设置插件配置"""
        self.plugin_config = config

    async def handle_raw_message(self, raw: Dict[str, Any]):
        """
        处理原始消息并转换为 MessageEnvelope
        
        Args:
            raw: OneBot 原始消息数据
            
        Returns:
            MessageEnvelope (dict)
        """
        from mofox_bus import MessageEnvelope, SegPayload, MessageInfoPayload, UserInfoPayload, GroupInfoPayload

        message_type = raw.get("message_type")
        message_id = str(raw.get("message_id", ""))
        message_time = time.time()
        
        # 构造用户信息
        sender_info = raw.get("sender", {})
        user_info: UserInfoPayload = {
            "platform": "qq",
            "user_id": str(sender_info.get("user_id", "")),
            "user_nickname": sender_info.get("nickname", ""),
            "user_cardname": sender_info.get("card", ""),
            "user_avatar": sender_info.get("avatar", ""),
        }

        # 构造群组信息（如果是群消息）
        group_info: Optional[GroupInfoPayload] = None
        if message_type == "group":
            group_id = raw.get("group_id")
            if group_id:
                group_info = {
                    "platform": "qq",
                    "group_id": str(group_id),
                    "group_name": "",  # 可以通过 API 获取
                }

        # 解析消息段
        message_segments = raw.get("message", [])
        seg_list: List[SegPayload] = []
        
        for seg in message_segments:
            seg_type = seg.get("type", "")
            seg_data = seg.get("data", {})
            
            # 转换为 SegPayload
            if seg_type == "text":
                seg_list.append({
                    "type": "text",
                    "data": seg_data.get("text", "")
                })
            elif seg_type == "image":
                # 这里需要下载图片并转换为 base64（简化版本）
                seg_list.append({
                    "type": "image",
                    "data": seg_data.get("url", "")  # 实际应该转换为 base64
                })
            elif seg_type == "at":
                seg_list.append({
                    "type": "at",
                    "data": f"{seg_data.get('qq', '')}"
                })
            # 其他消息类型...

        # 构造 MessageInfoPayload
        message_info = {
            "platform": "qq",
            "message_id": message_id,
            "time": message_time,
            "user_info": user_info,
            "format_info": {
                "content_format": ["text", "image"],  # 根据实际消息类型设置
                "accept_format": ["text", "image", "emoji", "voice"],
            },
        }
        
        # 添加群组信息（如果存在）
        if group_info:
            message_info["group_info"] = group_info

        # 构造 MessageEnvelope
        envelope = {
            "direction": "incoming",
            "message_info": message_info,
            "message_segment": {"type": "seglist", "data": seg_list} if len(seg_list) > 1 else (seg_list[0] if seg_list else {"type": "text", "data": ""}),
            "raw_message": raw.get("raw_message", ""),
            "platform": "qq",
            "message_id": message_id,
            "timestamp_ms": int(message_time * 1000),
        }

        return envelope



        

                

