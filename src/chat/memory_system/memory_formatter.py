"""记忆格式化工具

提供统一的记忆块格式化函数，供构建 Prompt 时使用。

当前使用的函数: format_memories_bracket_style
输入: list[dict] 其中每个元素包含：
    - display: str  记忆可读内容
    - memory_type: str  记忆类型 (personal_fact/opinion/preference/event 等)
    - metadata: dict 可选，包括
        - confidence: 置信度 (str|float)
        - importance: 重要度 (str|float)
        - timestamp: 时间戳 (float|str)
        - source: 来源 (str)
        - relevance_score: 相关度 (float)

返回: 适合直接嵌入提示词的大段文本；若无有效记忆返回空串。
"""
from __future__ import annotations

from typing import Any, Iterable
import time


def _format_timestamp(ts: Any) -> str:
    try:
        if ts in (None, ""):
            return ""
        if isinstance(ts, (int, float)) and ts > 0:
            return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(ts)))
        return str(ts)
    except Exception:
        return ""


def _coerce_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def format_memories_bracket_style(
    memories: Iterable[dict[str, Any]] | None,
    query_context: str | None = None,
    max_items: int = 15,
) -> str:
    """以方括号 + 标注字段的方式格式化记忆列表。

    例子输出:
        ## 相关记忆回顾
        - [类型:personal_fact|重要:高|置信:0.83|相关:0.72] 他喜欢黑咖啡 (来源: chat, 2025-10-05 09:30)

    Args:
        memories: 记忆字典迭代器
        query_context: 当前查询/用户的消息，用于在首行提示（可选）
        max_items: 最多输出的记忆条数
    Returns:
        str: 格式化文本；若无内容返回空串
    """
    if not memories:
        return ""

    lines: list[str] = ["## 相关记忆回顾"]
    if query_context:
        lines.append(f"（与当前消息相关：{query_context[:60]}{'...' if len(query_context) > 60 else ''}）")
    lines.append("")

    count = 0
    for mem in memories:
        if count >= max_items:
            break
        if not isinstance(mem, dict):
            continue
        display = _coerce_str(mem.get("display", "")).strip()
        if not display:
            continue

        mtype = _coerce_str(mem.get("memory_type", "fact")) or "fact"
        meta = mem.get("metadata", {}) if isinstance(mem.get("metadata"), dict) else {}
        confidence = _coerce_str(meta.get("confidence", ""))
        importance = _coerce_str(meta.get("importance", ""))
        source = _coerce_str(meta.get("source", ""))
        rel = meta.get("relevance_score")
        try:
            rel_str = f"{float(rel):.2f}" if rel is not None else ""
        except Exception:
            rel_str = ""
        ts = _format_timestamp(meta.get("timestamp"))

        # 构建标签段
        tags: list[str] = [f"类型:{mtype}"]
        if importance:
            tags.append(f"重要:{importance}")
        if confidence:
            tags.append(f"置信:{confidence}")
        if rel_str:
            tags.append(f"相关:{rel_str}")

        tag_block = "|".join(tags)
        suffix_parts = []
        if source:
            suffix_parts.append(source)
        if ts:
            suffix_parts.append(ts)
        suffix = (" (" + ", ".join(suffix_parts) + ")") if suffix_parts else ""

        lines.append(f"- [{tag_block}] {display}{suffix}")
        count += 1

    if count == 0:
        return ""

    if count >= max_items:
        lines.append(f"\n(已截断，仅显示前 {max_items} 条相关记忆)")

    return "\n".join(lines)


__all__ = ["format_memories_bracket_style"]
