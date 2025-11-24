"""
MessageEnvelope converter between mofox_bus schema and internal message structures.

- 优先处理 maim_message 风格的 message_info + message_segment。
- 兼容旧版 content/sender/channel 结构，方便逐步迁移。
""" 

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mofox_bus import (
    BaseMessageInfo,
    MessageBase,
    MessageEnvelope,
    Seg,
    UserInfo,
    GroupInfo,
)

from src.common.logger import get_logger

logger = get_logger("envelope_converter")


class EnvelopeConverter:
    """MessageEnvelope <-> MessageBase converter."""

    @staticmethod
    def to_message_base(envelope: MessageEnvelope) -> MessageBase:
        """
        Convert MessageEnvelope to MessageBase.
        """
        try:
            # 优先使用 maim_message 样式字段
            info_payload = envelope.get("message_info") or {}
            seg_payload = envelope.get("message_segment") or envelope.get("message_chain")

            if info_payload:
                message_info = BaseMessageInfo.from_dict(info_payload)
            else:
                message_info = EnvelopeConverter._build_info_from_legacy(envelope)

            if seg_payload is None:
                seg_list = EnvelopeConverter._content_to_segments(envelope.get("content"))
                seg_payload = seg_list

            message_segment = EnvelopeConverter._ensure_seg(seg_payload)
            raw_message = envelope.get("raw_message") or envelope.get("raw_platform_message")

            return MessageBase(
                message_info=message_info,
                message_segment=message_segment,
                raw_message=raw_message,
            )
        except Exception as e:
            logger.error(f"转换 MessageEnvelope 失败: {e}", exc_info=True)
            raise

    @staticmethod
    def _build_info_from_legacy(envelope: MessageEnvelope) -> BaseMessageInfo:
        """将 legacy 字段映射为 BaseMessageInfo。"""
        platform = envelope.get("platform")
        channel = envelope.get("channel") or {}
        sender = envelope.get("sender") or {}

        message_id = envelope.get("id") or envelope.get("message_id")
        timestamp_ms = envelope.get("timestamp_ms")
        time_value = (timestamp_ms / 1000.0) if timestamp_ms is not None else None

        group_info: Optional[GroupInfo] = None
        channel_type = channel.get("channel_type")
        if channel_type in ("group", "supergroup", "room"):
            group_info = GroupInfo(
                platform=platform,
                group_id=channel.get("channel_id"),
                group_name=channel.get("title"),
            )

        user_info: Optional[UserInfo] = None
        if sender:
            user_info = UserInfo(
                platform=platform,
                user_id=str(sender.get("user_id")) if sender.get("user_id") is not None else None,
                user_nickname=sender.get("display_name") or sender.get("user_nickname"),
                user_avatar=sender.get("avatar_url"),
            )

        return BaseMessageInfo(
            platform=platform,
            message_id=message_id,
            time=time_value,
            group_info=group_info,
            user_info=user_info,
            additional_config=envelope.get("metadata"),
        )

    @staticmethod
    def _ensure_seg(payload: Any) -> Seg:
        """将任意 payload 转为 Seg dataclass。"""
        if isinstance(payload, Seg):
            return payload
        if isinstance(payload, list):
            # 直接传入 Seg 列表或 seglist data
            return Seg(type="seglist", data=[EnvelopeConverter._ensure_seg(item) for item in payload])
        if isinstance(payload, dict):
            seg_type = payload.get("type") or "text"
            data = payload.get("data")
            if seg_type == "seglist" and isinstance(data, list):
                data = [EnvelopeConverter._ensure_seg(item) for item in data]
            return Seg(type=seg_type, data=data)
        # 兜底：转成文本片段
        return Seg(type="text", data="" if payload is None else str(payload))

    @staticmethod
    def _flatten_segments(seg: Seg) -> List[Seg]:
        """将 Seg/seglist 打平成列表，便于旧 content 转换。"""
        if seg.type == "seglist" and isinstance(seg.data, list):
            return [item if isinstance(item, Seg) else EnvelopeConverter._ensure_seg(item) for item in seg.data]
        return [seg]

    @staticmethod
    def _content_to_segments(content: Any) -> List[Seg]:
        """
        Convert legacy Content (type/data/metadata) to a flat list of Seg.
        """
        segments: List[Seg] = []

        def _walk(node: Any) -> None:
            if node is None:
                return
            if isinstance(node, list):
                for item in node:
                    _walk(item)
                return
            if not isinstance(node, dict):
                logger.warning("未知的 content 节点类型: %s", type(node))
                return

            content_type = node.get("type")
            data = node.get("data")
            metadata = node.get("metadata") or {}

            if content_type == "collection":
                items = data if isinstance(data, list) else node.get("items", [])
                for item in items:
                    _walk(item)
                return

            if content_type in ("text", "at"):
                subtype = metadata.get("subtype") or ("at" if content_type == "at" else None)
                text = "" if data is None else str(data)
                if subtype in ("at", "mention"):
                    user_info = metadata.get("user") or {}
                    seg_data: Dict[str, Any] = {
                        "user_id": user_info.get("id") or user_info.get("user_id"),
                        "user_name": user_info.get("name") or user_info.get("display_name"),
                        "text": text,
                        "raw": user_info.get("raw") or user_info if user_info else None,
                    }
                    if any(v is not None for v in seg_data.values()):
                        segments.append(Seg(type="at", data=seg_data))
                    else:
                        segments.append(Seg(type="at", data=text))
                else:
                    segments.append(Seg(type="text", data=text))
                return

            if content_type == "image":
                url = ""
                if isinstance(data, dict):
                    url = data.get("url") or data.get("file") or data.get("file_id") or ""
                elif data is not None:
                    url = str(data)
                segments.append(Seg(type="image", data=url))
                return

            if content_type == "audio":
                url = ""
                if isinstance(data, dict):
                    url = data.get("url") or data.get("file") or data.get("file_id") or ""
                elif data is not None:
                    url = str(data)
                segments.append(Seg(type="record", data=url))
                return

            if content_type == "video":
                url = ""
                if isinstance(data, dict):
                    url = data.get("url") or data.get("file") or data.get("file_id") or ""
                elif data is not None:
                    url = str(data)
                segments.append(Seg(type="video", data=url))
                return

            if content_type == "file":
                file_name = ""
                if isinstance(data, dict):
                    file_name = data.get("file_name") or data.get("name") or ""
                text = file_name or "[file]"
                segments.append(Seg(type="text", data=text))
                return

            if content_type == "command":
                name = ""
                args: Dict[str, Any] = {}
                if isinstance(data, dict):
                    name = data.get("name", "")
                    args = data.get("args", {}) or {}
                else:
                    name = str(data or "")
                cmd_text = f"/{name}" if name else "/command"
                if args:
                    cmd_text += " " + " ".join(f"{k}={v}" for k, v in args.items())
                segments.append(Seg(type="text", data=cmd_text))
                return

            if content_type == "event":
                event_type = ""
                if isinstance(data, dict):
                    event_type = data.get("event_type", "")
                else:
                    event_type = str(data or "")
                segments.append(Seg(type="text", data=f"[事件: {event_type or 'unknown'}]"))
                return

            if content_type == "system":
                text = "" if data is None else str(data)
                segments.append(Seg(type="text", data=f"[系统] {text}"))
                return

            logger.warning(f"未知的消息类型: {content_type}")
            segments.append(Seg(type="text", data=f"[未知消息类型: {content_type}]"))

        _walk(content)
        return segments

    @staticmethod
    def to_legacy_dict(envelope: MessageEnvelope) -> Dict[str, Any]:
        """
        Convert MessageEnvelope to legacy dict for backward compatibility.
        """
        message_base = EnvelopeConverter.to_message_base(envelope)
        return message_base.to_dict()

    @staticmethod
    def from_message_base(message: MessageBase, direction: str = "outgoing") -> MessageEnvelope:
        """
        Convert MessageBase to MessageEnvelope (maim_message style preferred).
        """
        try:
            info_dict = message.message_info.to_dict()
            seg_dict = message.message_segment.to_dict()

            envelope: MessageEnvelope = {
                "direction": direction,
                "message_info": info_dict,
                "message_segment": seg_dict,
                "platform": info_dict.get("platform"),
                "message_id": info_dict.get("message_id"),
                "schema_version": 1,
            }

            if message.message_info.time is not None:
                envelope["timestamp_ms"] = int(message.message_info.time * 1000)
            if message.raw_message is not None:
                envelope["raw_message"] = message.raw_message

            # legacy 补充，方便老代码继续工作
            segments = EnvelopeConverter._flatten_segments(message.message_segment)
            envelope["content"] = EnvelopeConverter._segments_to_content(segments)
            if message.message_info.user_info:
                envelope["sender"] = {
                    "user_id": message.message_info.user_info.user_id,
                    "role": "assistant" if direction == "outgoing" else "user",
                    "display_name": message.message_info.user_info.user_nickname,
                    "avatar_url": getattr(message.message_info.user_info, "user_avatar", None),
                }
            if message.message_info.group_info:
                envelope["channel"] = {
                    "channel_id": message.message_info.group_info.group_id,
                    "channel_type": "group",
                    "title": message.message_info.group_info.group_name,
                }

            return envelope

        except Exception as e:
            logger.error(f"转换 MessageBase 失败: {e}", exc_info=True)
            raise

    @staticmethod
    def _segments_to_content(segments: List[Seg]) -> Dict[str, Any]:
        """
        Convert Seg list to legacy Content (type/data/metadata).
        """
        if not segments:
            return {"type": "text", "data": ""}

        def _seg_to_content(seg: Seg) -> Dict[str, Any]:
            data = seg.data

            if seg.type == "text":
                return {"type": "text", "data": data}

            if seg.type == "at":
                content: Dict[str, Any] = {"type": "text", "data": ""}
                metadata: Dict[str, Any] = {"subtype": "at"}
                if isinstance(data, dict):
                    content["data"] = data.get("text", "")
                    user = {
                        "id": data.get("user_id"),
                        "name": data.get("user_name"),
                        "raw": data.get("raw"),
                    }
                    if any(v is not None for v in user.values()):
                        metadata["user"] = user
                else:
                    content["data"] = data
                if metadata:
                    content["metadata"] = metadata
                return content

            if seg.type == "image":
                return {"type": "image", "data": data}

            if seg.type in ("record", "voice", "audio"):
                return {"type": "audio", "data": data}

            if seg.type == "video":
                return {"type": "video", "data": data}

            return {"type": seg.type, "data": data}

        if len(segments) == 1:
            return _seg_to_content(segments[0])

        return {"type": "collection", "data": [_seg_to_content(seg) for seg in segments]}


__all__ = ["EnvelopeConverter"]
