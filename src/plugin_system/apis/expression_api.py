"""
表达方式管理API

提供表达方式的查询、创建、更新、删除功能
"""

import csv
import hashlib
import io
import math
import time
from typing import Any, Literal

import orjson
from sqlalchemy import and_, or_, select

from src.chat.express.expression_learner import ExpressionLearner
from src.chat.message_receive.chat_stream import get_chat_manager
from src.common.database.compatibility import get_db_session
from src.common.database.core.models import Expression
from src.common.database.optimization.cache_manager import get_cache
from src.common.database.utils.decorators import generate_cache_key
from src.common.logger import get_logger
from src.config.config import global_config

logger = get_logger("expression_api")


# ==================== 辅助函数 ====================


def parse_chat_id_input(chat_id_input: str) -> str:
    """
    解析聊天ID输入，支持两种格式：
    1. 哈希值格式（直接返回）
    2. platform:raw_id:type 格式（转换为哈希值）

    Args:
        chat_id_input: 输入的chat_id，可以是哈希值或 platform:raw_id:type 格式

    Returns:
        哈希值格式的chat_id

    Examples:
        >>> parse_chat_id_input("abc123def456")  # 哈希值
        "abc123def456"
        >>> parse_chat_id_input("QQ:12345:group")  # platform:id:type
        "..." (转换后的哈希值)
    """
    # 如果包含冒号，认为是 platform:id:type 格式
    if ":" in chat_id_input:
        parts = chat_id_input.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"无效的chat_id格式: {chat_id_input}，"
                "应为 'platform:raw_id:type' 格式，例如 'QQ:12345:group' 或 'QQ:67890:private'"
            )

        platform, raw_id, chat_type = parts

        if chat_type not in ["group", "private"]:
            raise ValueError(f"无效的chat_type: {chat_type}，只支持 'group' 或 'private'")

        # 使用与 ChatStream.get_stream_id 相同的逻辑生成哈希值
        is_group = chat_type == "group"
        components = [platform, raw_id] if is_group else [platform, raw_id, "private"]
        key = "_".join(components)
        return hashlib.sha256(key.encode()).hexdigest()

    # 否则认为已经是哈希值
    return chat_id_input


# ==================== 查询接口 ====================


async def get_expression_list(
    chat_id: str | None = None,
    type: Literal["style", "grammar"] | None = None,
    page: int = 1,
    page_size: int = 20,
    sort_by: Literal["count", "last_active_time", "create_date"] = "last_active_time",
    sort_order: Literal["asc", "desc"] = "desc",
) -> dict[str, Any]:
    """
    获取表达方式列表

    Args:
        chat_id: 聊天流ID，None表示获取所有
        type: 表达类型筛选
        page: 页码（从1开始）
        page_size: 每页数量
        sort_by: 排序字段
        sort_order: 排序顺序

    Returns:
        {
            "expressions": [...],
            "total": 100,
            "page": 1,
            "page_size": 20,
            "total_pages": 5
        }
    """
    try:
        async with get_db_session() as session:
            # 构建查询条件
            conditions = []
            if chat_id:
                conditions.append(Expression.chat_id == chat_id)
            if type:
                conditions.append(Expression.type == type)

            # 查询总数
            count_query = select(Expression)
            if conditions:
                count_query = count_query.where(and_(*conditions))
            count_result = await session.execute(count_query)
            total = len(list(count_result.scalars()))

            # 构建查询
            query = select(Expression)
            if conditions:
                query = query.where(and_(*conditions))

            # 排序
            sort_column = getattr(Expression, sort_by)
            if sort_order == "desc":
                query = query.order_by(sort_column.desc())
            else:
                query = query.order_by(sort_column.asc())

            # 分页
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)

            # 执行查询
            result = await session.execute(query)
            expressions = result.scalars().all()

            # 格式化结果
            expression_list = []
            chat_manager = get_chat_manager()

            for expr in expressions:
                # 获取聊天流名称和详细信息
                chat_name = await chat_manager.get_stream_name(expr.chat_id)
                chat_stream = await chat_manager.get_stream(expr.chat_id)

                # 构建格式化的chat_id信息
                chat_id_display = expr.chat_id  # 默认使用哈希值
                platform = "未知"
                raw_id = "未知"
                chat_type = "未知"

                if chat_stream:
                    platform = chat_stream.platform
                    if chat_stream.group_info:
                        raw_id = chat_stream.group_info.group_id
                        chat_type = "group"
                    elif chat_stream.user_info:
                        raw_id = chat_stream.user_info.user_id
                        chat_type = "private"
                    chat_id_display = f"{platform}:{raw_id}:{chat_type}"

                expression_list.append(
                    {
                        "id": expr.id,
                        "situation": expr.situation,
                        "style": expr.style,
                        "count": expr.count,
                        "last_active_time": expr.last_active_time,
                        "chat_id": expr.chat_id,  # 保留哈希值用于后端操作
                        "chat_id_display": chat_id_display,  # 显示用的格式化ID
                        "chat_platform": platform,
                        "chat_raw_id": raw_id,
                        "chat_type": chat_type,
                        "chat_name": chat_name or expr.chat_id,
                        "type": expr.type,
                        "create_date": expr.create_date if expr.create_date else expr.last_active_time,
                    }
                )

            total_pages = math.ceil(total / page_size) if total > 0 else 1

            return {
                "expressions": expression_list,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
            }

    except Exception as e:
        logger.error(f"获取表达方式列表失败: {e}")
        raise


async def get_expression_detail(expression_id: int) -> dict[str, Any] | None:
    """
    获取表达方式详情

    Returns:
        {
            "id": 1,
            "situation": "...",
            "style": "...",
            "count": 1.5,
            "last_active_time": 1234567890.0,
            "chat_id": "...",
            "type": "style",
            "create_date": 1234567890.0,
            "chat_name": "xxx群聊",
            "usage_stats": {...}
        }
    """
    try:
        async with get_db_session() as session:
            query = await session.execute(select(Expression).where(Expression.id == expression_id))
            expr = query.scalar()

            if not expr:
                return None

            # 获取聊天流名称和详细信息
            chat_manager = get_chat_manager()
            chat_name = await chat_manager.get_stream_name(expr.chat_id)
            chat_stream = await chat_manager.get_stream(expr.chat_id)

            # 构建格式化的chat_id信息
            chat_id_display = expr.chat_id
            platform = "未知"
            raw_id = "未知"
            chat_type = "未知"

            if chat_stream:
                platform = chat_stream.platform
                if chat_stream.group_info:
                    raw_id = chat_stream.group_info.group_id
                    chat_type = "group"
                elif chat_stream.user_info:
                    raw_id = chat_stream.user_info.user_id
                    chat_type = "private"
                chat_id_display = f"{platform}:{raw_id}:{chat_type}"

            # 计算使用统计
            days_since_create = (time.time() - (expr.create_date or expr.last_active_time)) / 86400
            days_since_last_use = (time.time() - expr.last_active_time) / 86400

            return {
                "id": expr.id,
                "situation": expr.situation,
                "style": expr.style,
                "count": expr.count,
                "last_active_time": expr.last_active_time,
                "chat_id": expr.chat_id,
                "chat_id_display": chat_id_display,
                "chat_platform": platform,
                "chat_raw_id": raw_id,
                "chat_type": chat_type,
                "chat_name": chat_name or expr.chat_id,
                "type": expr.type,
                "create_date": expr.create_date if expr.create_date else expr.last_active_time,
                "usage_stats": {
                    "days_since_create": round(days_since_create, 1),
                    "days_since_last_use": round(days_since_last_use, 1),
                    "usage_frequency": round(expr.count / max(days_since_create, 1), 3),
                },
            }

    except Exception as e:
        logger.error(f"获取表达方式详情失败: {e}")
        raise


async def search_expressions(
    keyword: str,
    search_field: Literal["situation", "style", "both"] = "both",
    chat_id: str | None = None,
    type: Literal["style", "grammar"] | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    搜索表达方式

    Args:
        keyword: 搜索关键词
        search_field: 搜索范围
        chat_id: 限定聊天流
        type: 限定类型
        limit: 最大返回数量
    """
    try:
        async with get_db_session() as session:
            # 构建搜索条件
            search_conditions = []
            if search_field in ["situation", "both"]:
                search_conditions.append(Expression.situation.contains(keyword))
            if search_field in ["style", "both"]:
                search_conditions.append(Expression.style.contains(keyword))

            # 构建其他条件
            other_conditions = []
            if chat_id:
                other_conditions.append(Expression.chat_id == chat_id)
            if type:
                other_conditions.append(Expression.type == type)

            # 组合查询
            query = select(Expression)
            if search_conditions:
                query = query.where(or_(*search_conditions))
            if other_conditions:
                query = query.where(and_(*other_conditions))

            query = query.order_by(Expression.count.desc()).limit(limit)

            # 执行查询
            result = await session.execute(query)
            expressions = result.scalars().all()

            # 格式化结果
            chat_manager = get_chat_manager()
            expression_list = []

            for expr in expressions:
                chat_name = await chat_manager.get_stream_name(expr.chat_id)
                expression_list.append(
                    {
                        "id": expr.id,
                        "situation": expr.situation,
                        "style": expr.style,
                        "count": expr.count,
                        "last_active_time": expr.last_active_time,
                        "chat_id": expr.chat_id,
                        "chat_name": chat_name or expr.chat_id,
                        "type": expr.type,
                        "create_date": expr.create_date if expr.create_date else expr.last_active_time,
                    }
                )

            return expression_list

    except Exception as e:
        logger.error(f"搜索表达方式失败: {e}")
        raise


async def get_expression_statistics(chat_id: str | None = None) -> dict[str, Any]:
    """
    获取表达方式统计信息

    Returns:
        {
            "total_count": 100,
            "style_count": 60,
            "grammar_count": 40,
            "top_used": [...],
            "recent_added": [...],
            "chat_distribution": {...}
        }
    """
    try:
        async with get_db_session() as session:
            # 构建基础查询
            base_query = select(Expression)
            if chat_id:
                base_query = base_query.where(Expression.chat_id == chat_id)

            # 总数
            all_result = await session.execute(base_query)
            all_expressions = list(all_result.scalars())
            total_count = len(all_expressions)

            # 按类型统计
            style_count = len([e for e in all_expressions if e.type == "style"])
            grammar_count = len([e for e in all_expressions if e.type == "grammar"])

            # Top 10 最常用
            top_used_query = base_query.order_by(Expression.count.desc()).limit(10)
            top_used_result = await session.execute(top_used_query)
            top_used_expressions = top_used_result.scalars().all()

            chat_manager = get_chat_manager()
            top_used = []
            for expr in top_used_expressions:
                chat_name = await chat_manager.get_stream_name(expr.chat_id)
                top_used.append(
                    {
                        "id": expr.id,
                        "situation": expr.situation,
                        "style": expr.style,
                        "count": expr.count,
                        "chat_name": chat_name or expr.chat_id,
                        "type": expr.type,
                    }
                )

            # 最近添加的10个
            recent_query = base_query.order_by(Expression.create_date.desc()).limit(10)
            recent_result = await session.execute(recent_query)
            recent_expressions = recent_result.scalars().all()

            recent_added = []
            for expr in recent_expressions:
                chat_name = await chat_manager.get_stream_name(expr.chat_id)
                recent_added.append(
                    {
                        "id": expr.id,
                        "situation": expr.situation,
                        "style": expr.style,
                        "count": expr.count,
                        "chat_name": chat_name or expr.chat_id,
                        "type": expr.type,
                        "create_date": expr.create_date if expr.create_date else expr.last_active_time,
                    }
                )

            # 按聊天流分布
            chat_distribution = {}
            for expr in all_expressions:
                chat_name = await chat_manager.get_stream_name(expr.chat_id)
                key = chat_name or expr.chat_id
                if key not in chat_distribution:
                    chat_distribution[key] = {"count": 0, "chat_id": expr.chat_id}
                chat_distribution[key]["count"] += 1

            return {
                "total_count": total_count,
                "style_count": style_count,
                "grammar_count": grammar_count,
                "top_used": top_used,
                "recent_added": recent_added,
                "chat_distribution": chat_distribution,
            }

    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise


# ==================== 管理接口 ====================


async def create_expression(
    situation: str, style: str, chat_id: str, type: Literal["style", "grammar"] = "style", count: float = 1.0
) -> dict[str, Any]:
    """
    手动创建表达方式

    Args:
        situation: 情境描述
        style: 表达风格
        chat_id: 聊天流ID，支持两种格式：
            - 哈希值格式（如: "abc123def456..."）
            - platform:raw_id:type 格式（如: "QQ:12345:group" 或 "QQ:67890:private"）
        type: 表达类型
        count: 权重

    Returns:
        创建的表达方式详情
    """
    try:
        # 解析并转换chat_id
        chat_id_hash = parse_chat_id_input(chat_id)
        current_time = time.time()

        async with get_db_session() as session:
            # 检查是否已存在
            existing_query = await session.execute(
                select(Expression).where(
                    and_(
                        Expression.chat_id == chat_id_hash,
                        Expression.type == type,
                        Expression.situation == situation,
                        Expression.style == style,
                    )
                )
            )
            existing = existing_query.scalar()

            if existing:
                raise ValueError("该表达方式已存在")

            # 创建新表达方式
            new_expression = Expression(
                situation=situation,
                style=style,
                count=count,
                last_active_time=current_time,
                chat_id=chat_id_hash,
                type=type,
                create_date=current_time,
            )

            session.add(new_expression)
            await session.commit()
            await session.refresh(new_expression)

            # 清除缓存
            cache = await get_cache()
            await cache.delete(generate_cache_key("chat_expressions", chat_id_hash))

            logger.info(f"创建表达方式成功: {situation} -> {style} (chat_id={chat_id_hash})")

            return await get_expression_detail(new_expression.id)  # type: ignore

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"创建表达方式失败: {e}")
        raise


async def update_expression(
    expression_id: int,
    situation: str | None = None,
    style: str | None = None,
    count: float | None = None,
    type: Literal["style", "grammar"] | None = None,
) -> bool:
    """
    更新表达方式

    Returns:
        是否成功
    """
    try:
        async with get_db_session() as session:
            query = await session.execute(select(Expression).where(Expression.id == expression_id))
            expr = query.scalar()

            if not expr:
                return False

            # 更新字段
            if situation is not None:
                expr.situation = situation
            if style is not None:
                expr.style = style
            if count is not None:
                expr.count = max(0.0, min(5.0, count))  # 限制在0-5之间
            if type is not None:
                expr.type = type

            expr.last_active_time = time.time()

            await session.commit()

            # 清除缓存
            cache = await get_cache()
            await cache.delete(generate_cache_key("chat_expressions", expr.chat_id))

            logger.info(f"更新表达方式成功: ID={expression_id}")
            return True

    except Exception as e:
        logger.error(f"更新表达方式失败: {e}")
        raise


async def delete_expression(expression_id: int) -> bool:
    """
    删除表达方式
    """
    try:
        async with get_db_session() as session:
            query = await session.execute(select(Expression).where(Expression.id == expression_id))
            expr = query.scalar()

            if not expr:
                return False

            chat_id = expr.chat_id
            await session.delete(expr)
            await session.commit()

            # 清除缓存
            cache = await get_cache()
            await cache.delete(generate_cache_key("chat_expressions", chat_id))

            logger.info(f"删除表达方式成功: ID={expression_id}")
            return True

    except Exception as e:
        logger.error(f"删除表达方式失败: {e}")
        raise


async def batch_delete_expressions(expression_ids: list[int]) -> int:
    """
    批量删除表达方式

    Returns:
        删除的数量
    """
    try:
        deleted_count = 0
        affected_chat_ids = set()

        async with get_db_session() as session:
            for expr_id in expression_ids:
                query = await session.execute(select(Expression).where(Expression.id == expr_id))
                expr = query.scalar()

                if expr:
                    affected_chat_ids.add(expr.chat_id)
                    await session.delete(expr)
                    deleted_count += 1

            await session.commit()

        # 清除缓存
        cache = await get_cache()
        for chat_id in affected_chat_ids:
            await cache.delete(generate_cache_key("chat_expressions", chat_id))

        logger.info(f"批量删除表达方式成功: 删除了 {deleted_count} 个")
        return deleted_count

    except Exception as e:
        logger.error(f"批量删除表达方式失败: {e}")
        raise


async def activate_expression(expression_id: int, increment: float = 0.1) -> bool:
    """
    激活表达方式（增加权重）
    """
    try:
        async with get_db_session() as session:
            query = await session.execute(select(Expression).where(Expression.id == expression_id))
            expr = query.scalar()

            if not expr:
                return False

            # 增加count，但不超过5.0
            expr.count = min(expr.count + increment, 5.0)
            expr.last_active_time = time.time()

            await session.commit()

            # 清除缓存
            cache = await get_cache()
            await cache.delete(generate_cache_key("chat_expressions", expr.chat_id))

            logger.info(f"激活表达方式成功: ID={expression_id}, new count={expr.count:.2f}")
            return True

    except Exception as e:
        logger.error(f"激活表达方式失败: {e}")
        raise


# ==================== 学习管理接口 ====================

async def get_learning_status(chat_id: str) -> dict[str, Any]:
    """
    获取学习状态

    Args:
        chat_id: 聊天流ID，支持两种格式：
            - 哈希值格式（如: "abc123def456..."）
            - platform:raw_id:type 格式（如: "QQ:12345:group" 或 "QQ:67890:private"）

    Returns:
        {
            "can_learn": true,
            "enable_learning": true,
            "learning_intensity": 1.0,
            "last_learning_time": 1234567890.0,
            "messages_since_last": 25,
            "next_learning_in": 180.0
        }
    """
    try:
        # 解析并转换chat_id
        chat_id_hash = parse_chat_id_input(chat_id)

        learner = ExpressionLearner(chat_id_hash)
        await learner._initialize_chat_name()

        # 获取配置
        if global_config is None:
            raise RuntimeError("Global config is not initialized")

        _use_expression, enable_learning, learning_intensity = global_config.expression.get_expression_config_for_chat(
            chat_id_hash
        )

        can_learn = learner.can_learn_for_chat()
        should_trigger = await learner.should_trigger_learning()

        # 计算距离下次学习的时间
        min_interval = learner.min_learning_interval / learning_intensity
        time_since_last = time.time() - learner.last_learning_time
        next_learning_in = max(0, min_interval - time_since_last)

        # 获取消息统计
        from src.chat.utils.chat_message_builder import get_raw_msg_by_timestamp_with_chat_inclusive

        recent_messages = await get_raw_msg_by_timestamp_with_chat_inclusive(
            chat_id=chat_id_hash,
            timestamp_start=learner.last_learning_time,
            timestamp_end=time.time(),
            filter_bot=True,
        )
        messages_since_last = len(recent_messages) if recent_messages else 0

        return {
            "can_learn": can_learn,
            "enable_learning": enable_learning,
            "learning_intensity": learning_intensity,
            "last_learning_time": learner.last_learning_time,
            "messages_since_last": messages_since_last,
            "next_learning_in": next_learning_in,
            "should_trigger": should_trigger,
            "min_messages_required": learner.min_messages_for_learning,
        }

    except Exception as e:
        logger.error(f"获取学习状态失败: {e}")
        raise


# ==================== 共享组管理接口 ====================


async def get_sharing_groups() -> list[dict[str, Any]]:
    """
    获取所有共享组配置

    Returns:
        [
            {
                "group_name": "group_a",
                "chat_streams": [...],
                "expression_count": 50
            },
            ...
        ]
    """
    try:
        if global_config is None:
            return []

        groups: dict[str, dict] = {}
        chat_manager = get_chat_manager()

        for rule in global_config.expression.rules:
            if rule.group and rule.chat_stream_id:
                # 解析chat_id
                from src.chat.express.expression_learner import ExpressionLearner

                chat_id = ExpressionLearner._parse_stream_config_to_chat_id(rule.chat_stream_id)

                if not chat_id:
                    continue

                if rule.group not in groups:
                    groups[rule.group] = {"group_name": rule.group, "chat_streams": [], "expression_count": 0}

                # 获取聊天流名称
                chat_name = await chat_manager.get_stream_name(chat_id)

                groups[rule.group]["chat_streams"].append(
                    {
                        "chat_id": chat_id,
                        "chat_name": chat_name or chat_id,
                        "stream_config": rule.chat_stream_id,
                        "learn_expression": rule.learn_expression,
                        "use_expression": rule.use_expression,
                    }
                )

        # 统计每个组的表达方式数量
        async with get_db_session() as session:
            for group_data in groups.values():
                chat_ids = [stream["chat_id"] for stream in group_data["chat_streams"]]
                if chat_ids:
                    query = await session.execute(select(Expression).where(Expression.chat_id.in_(chat_ids)))
                    expressions = list(query.scalars())
                    group_data["expression_count"] = len(expressions)

        return list(groups.values())

    except Exception as e:
        logger.error(f"获取共享组失败: {e}")
        raise


async def get_related_chat_ids(chat_id: str) -> list[str]:
    """
    获取与指定聊天流共享表达方式的所有聊天流ID
    """
    try:
        learner = ExpressionLearner(chat_id)
        related_ids = learner.get_related_chat_ids()

        # 获取每个聊天流的名称
        chat_manager = get_chat_manager()
        result = []

        for cid in related_ids:
            chat_name = await chat_manager.get_stream_name(cid)
            result.append({"chat_id": cid, "chat_name": chat_name or cid})

        return result

    except Exception as e:
        logger.error(f"获取关联聊天流失败: {e}")
        raise


# ==================== 导入导出接口 ====================


async def export_expressions(
    chat_id: str | None = None, type: Literal["style", "grammar"] | None = None, format: Literal["json", "csv"] = "json"
) -> str:
    """
    导出表达方式

    Returns:
        导出的文件内容（JSON字符串或CSV文本）
    """
    try:
        async with get_db_session() as session:
            # 构建查询
            query = select(Expression)
            conditions = []
            if chat_id:
                conditions.append(Expression.chat_id == chat_id)
            if type:
                conditions.append(Expression.type == type)

            if conditions:
                query = query.where(and_(*conditions))

            result = await session.execute(query)
            expressions = result.scalars().all()

            if format == "json":
                # JSON格式
                data = [
                    {
                        "situation": expr.situation,
                        "style": expr.style,
                        "count": expr.count,
                        "last_active_time": expr.last_active_time,
                        "chat_id": expr.chat_id,
                        "type": expr.type,
                        "create_date": expr.create_date if expr.create_date else expr.last_active_time,
                    }
                    for expr in expressions
                ]
                return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()

            else:  # csv
                # CSV格式
                output = io.StringIO()
                writer = csv.writer(output)

                # 写入标题
                writer.writerow(["situation", "style", "count", "last_active_time", "chat_id", "type", "create_date"])

                # 写入数据
                for expr in expressions:
                    writer.writerow(
                        [
                            expr.situation,
                            expr.style,
                            expr.count,
                            expr.last_active_time,
                            expr.chat_id,
                            expr.type,
                            expr.create_date if expr.create_date else expr.last_active_time,
                        ]
                    )

                return output.getvalue()

    except Exception as e:
        logger.error(f"导出表达方式失败: {e}")
        raise


async def import_expressions(
    data: str,
    format: Literal["json", "csv"] = "json",
    chat_id: str | None = None,
    merge_strategy: Literal["skip", "replace", "merge"] = "skip",
) -> dict[str, Any]:
    """
    导入表达方式

    Args:
        data: 导入数据
        format: 数据格式
        chat_id: 目标聊天流ID，None表示使用原chat_id
        merge_strategy:
            - skip: 跳过已存在的
            - replace: 替换已存在的
            - merge: 合并（累加count）

    Returns:
        {
            "imported": 10,
            "skipped": 2,
            "replaced": 1,
            "errors": []
        }
    """
    try:
        imported_count = 0
        skipped_count = 0
        replaced_count = 0
        errors = []

        # 解析数据
        if format == "json":
            try:
                expressions_data = orjson.loads(data)
            except Exception as e:
                raise ValueError(f"无效的JSON格式: {e}")
        else:  # csv
            try:
                reader = csv.DictReader(io.StringIO(data))
                expressions_data = list(reader)
            except Exception as e:
                raise ValueError(f"无效的CSV格式: {e}")

        # 导入表达方式
        async with get_db_session() as session:
            affected_chat_ids = set()

            for idx, expr_data in enumerate(expressions_data):
                try:
                    # 提取字段
                    situation = expr_data.get("situation", "").strip()
                    style = expr_data.get("style", "").strip()
                    count = float(expr_data.get("count", 1.0))
                    expr_type = expr_data.get("type", "style")
                    target_chat_id = chat_id if chat_id else expr_data.get("chat_id")

                    if not situation or not style or not target_chat_id:
                        errors.append(f"行 {idx + 1}: 缺少必要字段")
                        continue

                    # 检查是否已存在
                    existing_query = await session.execute(
                        select(Expression).where(
                            and_(
                                Expression.chat_id == target_chat_id,
                                Expression.type == expr_type,
                                Expression.situation == situation,
                                Expression.style == style,
                            )
                        )
                    )
                    existing = existing_query.scalar()

                    if existing:
                        if merge_strategy == "skip":
                            skipped_count += 1
                            continue
                        elif merge_strategy == "replace":
                            existing.count = count
                            existing.last_active_time = time.time()
                            replaced_count += 1
                            affected_chat_ids.add(target_chat_id)
                        elif merge_strategy == "merge":
                            existing.count = min(existing.count + count, 5.0)
                            existing.last_active_time = time.time()
                            replaced_count += 1
                            affected_chat_ids.add(target_chat_id)
                    else:
                        # 创建新的
                        current_time = time.time()
                        new_expr = Expression(
                            situation=situation,
                            style=style,
                            count=min(count, 5.0),
                            last_active_time=current_time,
                            chat_id=target_chat_id,
                            type=expr_type,
                            create_date=current_time,
                        )
                        session.add(new_expr)
                        imported_count += 1
                        affected_chat_ids.add(target_chat_id)

                except Exception as e:
                    errors.append(f"行 {idx + 1}: {e!s}")

            await session.commit()

        # 清除缓存
        cache = await get_cache()
        for cid in affected_chat_ids:
            await cache.delete(generate_cache_key("chat_expressions", cid))

        logger.info(
            f"导入完成: 导入{imported_count}个, 跳过{skipped_count}个, "
            f"替换{replaced_count}个, 错误{len(errors)}个"
        )

        return {"imported": imported_count, "skipped": skipped_count, "replaced": replaced_count, "errors": errors}

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"导入表达方式失败: {e}")
        raise
