"""特殊消息解析器模块

解析 QQ/OneBot 协议中的特殊消息类型：
- XML 消息：群公告、分享卡片等
- JSON 消息：小程序分享、音乐分享、天气分享等
- Location 消息：位置分享
- Share 消息：网页链接分享
- Contact 消息：QQ/群名片推荐

这些消息通常包含结构化数据，需要提取关键信息转换为可读文本。
"""

import re
import xml.etree.ElementTree as ET
from typing import Any

import orjson

from src.common.logger import get_logger

logger = get_logger("special_message_parser")


# =============================================================================
# XML 消息解析
# =============================================================================


def parse_xml_message(xml_data: str | dict) -> str:
    """解析 XML 消息

    XML 消息常见于：
    - 群公告
    - 红包（旧版）
    - 第三方分享卡片
    - 位置分享（旧格式）

    Args:
        xml_data: XML 字符串或包含 data 字段的字典

    Returns:
        str: 解析后的可读文本
    """
    # 提取 XML 字符串
    if isinstance(xml_data, dict):
        xml_content = xml_data.get("data", "")
    else:
        xml_content = xml_data

    if not xml_content:
        return "[XML消息]"

    try:
        # 尝试解析 XML
        root = ET.fromstring(xml_content)

        # 提取常用属性
        brief = root.get("brief", "")
        service_id = root.get("serviceID", "")

        # 提取标题和摘要
        title_elem = root.find(".//title")
        summary_elem = root.find(".//summary")
        source_elem = root.find(".//source")

        title = title_elem.text if title_elem is not None and title_elem.text else ""
        summary = summary_elem.text if summary_elem is not None and summary_elem.text else ""
        source = source_elem.get("name", "") if source_elem is not None else ""

        # 构建可读文本
        parts = []

        # 优先使用 brief 作为消息类型提示
        if brief:
            parts.append(f"[{brief}]")
        elif service_id:
            parts.append("[XML卡片消息]")
        else:
            parts.append("[XML消息]")

        if title:
            parts.append(f"标题：{title}")

        if summary:
            # 限制摘要长度
            if len(summary) > 200:
                summary = summary[:200] + "..."
            parts.append(f"内容：{summary}")

        if source:
            parts.append(f"来源：{source}")

        # 尝试提取 URL
        url = root.get("url", "")
        if not url:
            url_elem = root.find(".//url")
            if url_elem is not None and url_elem.text:
                url = url_elem.text
        if url:
            parts.append(f"链接：{url}")

        return " ".join(parts) if len(parts) > 1 else parts[0] if parts else "[XML消息]"

    except ET.ParseError as e:
        logger.warning(f"XML 解析失败: {e}, 原始数据: {xml_content[:200]}...")
        # 尝试使用正则提取关键信息
        return _fallback_xml_parse(xml_content)
    except Exception as e:
        logger.error(f"XML 消息处理异常: {e}")
        return "[XML消息]"


def _fallback_xml_parse(xml_content: str) -> str:
    """XML 解析失败时的回退方案，使用正则提取关键信息"""
    parts = ["[XML消息]"]

    # 提取 brief
    brief_match = re.search(r'brief="([^"]*)"', xml_content)
    if brief_match:
        parts[0] = f"[{brief_match.group(1)}]"

    # 提取 title
    title_match = re.search(r"<title>([^<]+)</title>", xml_content)
    if title_match:
        parts.append(f"标题：{title_match.group(1)}")

    # 提取 summary
    summary_match = re.search(r"<summary>([^<]+)</summary>", xml_content)
    if summary_match:
        summary = summary_match.group(1)
        if len(summary) > 200:
            summary = summary[:200] + "..."
        parts.append(f"内容：{summary}")

    return " ".join(parts)


# =============================================================================
# JSON 消息解析
# =============================================================================


def parse_json_message(json_data: str | dict) -> str:
    """解析 JSON 消息

    JSON 消息常见于：
    - QQ 小程序分享（B站、知乎、微博等）
    - 音乐分享（QQ音乐、网易云音乐）
    - 天气分享
    - 游戏分享
    - 直播间卡片
    - ARK 卡片消息

    Args:
        json_data: JSON 字符串或字典

    Returns:
        str: 解析后的可读文本
    """
    # 解析 JSON
    if isinstance(json_data, str):
        try:
            data = orjson.loads(json_data)
        except Exception as e:
            logger.warning(f"JSON 解析失败: {e}")
            return "[JSON消息]"
    elif isinstance(json_data, dict):
        # 可能是 {"data": "json_string"} 格式
        inner_data = json_data.get("data", json_data)
        if isinstance(inner_data, str):
            try:
                data = orjson.loads(inner_data)
            except Exception:
                return "[JSON消息]"
        else:
            data = inner_data
    else:
        return "[JSON消息]"

    if not isinstance(data, dict):
        return "[JSON消息]"

    # 获取 app 标识来确定消息类型
    app = data.get("app", "")
    prompt = data.get("prompt", "")

    # 根据 app 类型分发处理
    try:
        if "miniapp" in app.lower() or "小程序" in prompt:
            return _parse_miniapp_message(data)

        elif "music" in app.lower() or "音乐" in prompt:
            return _parse_music_message(data)

        elif "weather" in app.lower() or "天气" in prompt:
            return _parse_weather_message(data)

        elif "map" in app.lower() or "位置" in prompt:
            return _parse_map_message(data)

        elif "contact" in app.lower() or "名片" in prompt:
            return _parse_contact_message(data)

        elif "gamecenter" in app.lower() or "游戏" in prompt:
            return _parse_game_message(data)

        elif "structmsg" in app.lower():
            return _parse_structmsg_message(data)

        else:
            # 通用解析
            return _parse_generic_json_message(data)

    except Exception as e:
        logger.error(f"JSON 消息解析异常: {e}, app: {app}")
        return _parse_generic_json_message(data)


def _parse_miniapp_message(data: dict) -> str:
    """解析小程序分享消息"""
    meta = data.get("meta", {})
    prompt = data.get("prompt", "[小程序]")

    # 尝试从 meta 中提取详情
    # 常见的 meta 结构: detail_1, news, etc.
    detail = None
    for key in meta:
        if isinstance(meta[key], dict):
            detail = meta[key]
            break

    if detail:
        title = detail.get("title", detail.get("desc", ""))
        desc = detail.get("desc", detail.get("preview", ""))
        source = detail.get("source", detail.get("tag", ""))

        parts = [prompt if prompt else "[小程序分享]"]
        if title:
            parts.append(f"「{title}」")
        if desc and desc != title:
            if len(desc) > 100:
                desc = desc[:100] + "..."
            parts.append(desc)
        if source:
            parts.append(f"— {source}")

        return " ".join(parts)

    # 回退到通用解析
    return _parse_generic_json_message(data)


def _parse_music_message(data: dict) -> str:
    """解析音乐分享消息"""
    meta = data.get("meta", {})
    prompt = data.get("prompt", "[音乐]")

    # 查找音乐信息
    music_info = meta.get("music", {})
    if not music_info:
        # 尝试其他可能的 key
        for key in meta:
            if isinstance(meta[key], dict) and "musicUrl" in meta[key]:
                music_info = meta[key]
                break

    title = music_info.get("title", "")
    desc = music_info.get("desc", "")
    source = music_info.get("source", music_info.get("tag", ""))

    parts = [prompt if prompt else "[音乐分享]"]
    if title:
        parts.append(f"🎵「{title}」")
    if desc:
        parts.append(f"- {desc}")
    if source:
        parts.append(f"来自 {source}")

    return " ".join(parts) if len(parts) > 1 else "[音乐分享]"


def _parse_weather_message(data: dict) -> str:
    """解析天气分享消息"""
    meta = data.get("meta", {})
    prompt = data.get("prompt", "[天气]")

    weather_info = None
    for key in meta:
        if isinstance(meta[key], dict):
            weather_info = meta[key]
            break

    if weather_info:
        city = weather_info.get("city", weather_info.get("title", ""))
        weather = weather_info.get("weather", weather_info.get("desc", ""))
        temp = weather_info.get("temp", weather_info.get("temperature", ""))

        parts = ["[天气分享]"]
        if city:
            parts.append(f"📍 {city}")
        if weather:
            parts.append(f"🌤️ {weather}")
        if temp:
            parts.append(f"🌡️ {temp}")

        return " ".join(parts)

    return prompt if prompt else "[天气分享]"


def _parse_map_message(data: dict) -> str:
    """解析地图/位置分享消息"""
    meta = data.get("meta", {})
    prompt = data.get("prompt", "[位置]")

    location_info = None
    for key in meta:
        if isinstance(meta[key], dict):
            location_info = meta[key]
            break

    if location_info:
        name = location_info.get("name", location_info.get("title", ""))
        address = location_info.get("address", location_info.get("desc", ""))

        parts = ["[位置分享]"]
        if name:
            parts.append(f"📍 {name}")
        if address and address != name:
            parts.append(f"地址：{address}")

        return " ".join(parts)

    return prompt if prompt else "[位置分享]"


def _parse_contact_message(data: dict) -> str:
    """解析联系人/名片分享消息"""
    meta = data.get("meta", {})
    prompt = data.get("prompt", "[名片]")

    contact_info = None
    for key in meta:
        if isinstance(meta[key], dict):
            contact_info = meta[key]
            break

    if contact_info:
        name = contact_info.get("name", contact_info.get("nickname", ""))
        qq = contact_info.get("uin", contact_info.get("qq", ""))

        if name:
            return f"[推荐名片] {name}" + (f" (QQ: {qq})" if qq else "")

    return prompt if prompt else "[推荐名片]"


def _parse_game_message(data: dict) -> str:
    """解析游戏分享消息"""
    meta = data.get("meta", {})
    prompt = data.get("prompt", "[游戏]")

    game_info = None
    for key in meta:
        if isinstance(meta[key], dict):
            game_info = meta[key]
            break

    if game_info:
        title = game_info.get("title", game_info.get("name", ""))
        desc = game_info.get("desc", "")

        parts = ["[游戏分享]"]
        if title:
            parts.append(f"🎮「{title}」")
        if desc:
            if len(desc) > 100:
                desc = desc[:100] + "..."
            parts.append(desc)

        return " ".join(parts)

    return prompt if prompt else "[游戏分享]"


def _parse_structmsg_message(data: dict) -> str:
    """解析结构化消息"""
    meta = data.get("meta", {})
    prompt = data.get("prompt", "")
    desc = data.get("desc", "")

    # 尝试从 meta 中提取新闻/文章信息
    news = meta.get("news", {})
    if news:
        title = news.get("title", "")
        desc_text = news.get("desc", news.get("preview", ""))
        source = news.get("tag", news.get("source", ""))

        parts = ["[卡片消息]"]
        if title:
            parts.append(f"「{title}」")
        if desc_text:
            if len(desc_text) > 100:
                desc_text = desc_text[:100] + "..."
            parts.append(desc_text)
        if source:
            parts.append(f"— {source}")

        return " ".join(parts)

    # 回退
    if prompt:
        return prompt
    if desc:
        return f"[卡片消息] {desc}"
    return "[卡片消息]"


def _parse_generic_json_message(data: dict) -> str:
    """通用 JSON 消息解析（回退方案）"""
    prompt = data.get("prompt", "")
    desc = data.get("desc", "")

    # 尝试从 meta 中提取任何有用信息
    meta = data.get("meta", {})
    title = ""
    detail_desc = ""

    for value in meta.values():
        if isinstance(value, dict):
            if not title:
                title = value.get("title", value.get("name", ""))
            if not detail_desc:
                detail_desc = value.get("desc", value.get("preview", value.get("summary", "")))
            if title and detail_desc:
                break

    # 构建输出
    parts = []

    if prompt:
        parts.append(prompt)
    else:
        parts.append("[卡片消息]")

    if title:
        parts.append(f"「{title}」")

    if detail_desc and detail_desc != title:
        if len(detail_desc) > 100:
            detail_desc = detail_desc[:100] + "..."
        parts.append(detail_desc)
    elif desc and desc not in parts:
        if len(desc) > 100:
            desc = desc[:100] + "..."
        parts.append(desc)

    return " ".join(parts)


# =============================================================================
# Location 消息解析
# =============================================================================


def parse_location_message(location_data: dict) -> str:
    """解析位置消息

    OneBot 格式：
    {
        "type": "location",
        "data": {
            "lat": "39.8969426",
            "lon": "116.3109099",
            "title": "位置名称",
            "content": "详细地址"
        }
    }

    Args:
        location_data: 位置消息数据

    Returns:
        str: 解析后的可读文本
    """
    if isinstance(location_data, dict):
        data = location_data.get("data", location_data)
    else:
        return "[位置消息]"

    title = data.get("title", data.get("name", ""))
    content = data.get("content", data.get("address", ""))
    lat = data.get("lat", data.get("latitude", ""))
    lon = data.get("lon", data.get("longitude", ""))

    parts = ["[位置分享]"]

    if title:
        parts.append(f"📍 {title}")

    if content and content != title:
        parts.append(f"地址：{content}")

    if lat and lon:
        parts.append(f"坐标：({lat}, {lon})")

    return " ".join(parts) if len(parts) > 1 else "[位置消息]"


# =============================================================================
# Share 消息解析
# =============================================================================


def parse_share_message(share_data: dict) -> str:
    """解析链接分享消息

    OneBot 格式：
    {
        "type": "share",
        "data": {
            "url": "http://example.com",
            "title": "分享标题",
            "content": "分享内容描述",
            "image": "https://example.com/preview.jpg"
        }
    }

    Args:
        share_data: 分享消息数据

    Returns:
        str: 解析后的可读文本
    """
    if isinstance(share_data, dict):
        data = share_data.get("data", share_data)
    else:
        return "[链接分享]"

    url = data.get("url", "")
    title = data.get("title", "")
    content = data.get("content", data.get("desc", ""))

    parts = ["[链接分享]"]

    if title:
        parts.append(f"「{title}」")

    if content:
        if len(content) > 100:
            content = content[:100] + "..."
        parts.append(content)

    if url:
        parts.append(f"🔗 {url}")

    return " ".join(parts) if len(parts) > 1 else "[链接分享]"


# =============================================================================
# Contact 消息解析
# =============================================================================


def parse_contact_message(contact_data: dict) -> str:
    """解析推荐名片消息

    OneBot 格式：
    {
        "type": "contact",
        "data": {
            "type": "qq",  // 或 "group"
            "id": "10001000"
        }
    }

    Args:
        contact_data: 名片消息数据

    Returns:
        str: 解析后的可读文本
    """
    if isinstance(contact_data, dict):
        data = contact_data.get("data", contact_data)
    else:
        return "[推荐名片]"

    contact_type = data.get("type", "qq")
    contact_id = data.get("id", "")

    if contact_type == "group":
        return f"[推荐群名片] 群号：{contact_id}" if contact_id else "[推荐群名片]"
    else:
        return f"[推荐QQ名片] QQ：{contact_id}" if contact_id else "[推荐QQ名片]"


# =============================================================================
# Forward 消息解析（合并转发）
# =============================================================================


def parse_forward_message(forward_data: dict) -> str:
    """解析合并转发消息

    转发消息通常包含多条消息，需要递归处理。
    这里只做简单的标记，具体内容的获取需要调用 API。

    Args:
        forward_data: 转发消息数据

    Returns:
        str: 解析后的可读文本
    """
    if isinstance(forward_data, dict):
        data = forward_data.get("data", forward_data)
    else:
        return "[合并转发消息]"

    # 转发消息可能包含 id 用于获取详细内容
    forward_id = data.get("id", data.get("resid", ""))

    if forward_id:
        return f"[合并转发消息] ID: {forward_id}"

    return "[合并转发消息]"


# =============================================================================
# 统一入口
# =============================================================================


def parse_special_message(seg_type: str, seg_data: Any) -> str | None:
    """统一的特殊消息解析入口

    Args:
        seg_type: 消息段类型
        seg_data: 消息段数据

    Returns:
        str | None: 解析后的可读文本，如果不是特殊消息类型则返回 None
    """
    parsers = {
        "xml": parse_xml_message,
        "json": parse_json_message,
        "location": parse_location_message,
        "share": parse_share_message,
        "contact": parse_contact_message,
        "forward": parse_forward_message,
    }

    parser = parsers.get(seg_type)
    if parser:
        try:
            return parser(seg_data)
        except Exception as e:
            logger.error(f"解析 {seg_type} 消息失败: {e}")
            return f"[{seg_type}消息]"

    return None
