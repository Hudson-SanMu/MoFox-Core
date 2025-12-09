import traceback
from collections import defaultdict
from typing import Any

from sqlalchemy import func, not_, select
from sqlalchemy.orm import DeclarativeBase

from src.common.database.compatibility import get_db_session

# from src.common.database.database_model import Messages
from src.common.database.core.models import Messages
from src.common.logger import get_logger
from src.config.config import global_config

logger = get_logger(__name__)


SAFE_FETCH_LIMIT = 5000  # 防止一次性读取过多行导致内存暴涨


class Base(DeclarativeBase):
    pass


def _model_to_dict(instance: Base) -> dict[str, Any]:
    """
    将 SQLAlchemy 模型实例转换为字典。
    """
    try:
        return {col.name: getattr(instance, col.name) for col in instance.__table__.columns}
    except Exception as e:
        # 如果对象已经脱离会话，尝试从instance.__dict__中获取数据
        logger.warning(f"从数据库对象获取属性失败，尝试使用__dict__: {e}")
        return {col.name: instance.__dict__.get(col.name) for col in instance.__table__.columns}


async def find_messages(
    message_filter: dict[str, Any],
    sort: list[tuple[str, int]] | None = None,
    limit: int = 0,
    limit_mode: str = "latest",
    filter_bot=False,
    filter_command=False,
    filter_meaningless=False,
) -> list[dict[str, Any]]:
    """
    根据提供的过滤器、排序和限制条件查找消息。

    Args:
        message_filter: 查询过滤器字典，键为模型字段名，值为期望值或包含操作符的字典 (例如 {'$gt': value}).
        sort: 排序条件列表，例如 [('time', 1)] (1 for asc, -1 for desc)。仅在 limit 为 0 时生效。
        limit: 返回的最大文档数，0表示不限制。
        limit_mode: 当 limit > 0 时生效。 'earliest' 表示获取最早的记录， 'latest' 表示获取最新的记录（结果仍按时间正序排列）。默认为 'latest'。
        filter_meaningless: 是否过滤无意义消息（表情包、通知、纯回复等）。用于表达学习等场景。

    Returns:
        消息字典列表，如果出错则返回空列表。
    """
    try:
        assert global_config is not None
        async with get_db_session() as session:
            query = select(Messages)

            # 应用过滤器
            if message_filter:
                conditions = []
                for key, value in message_filter.items():
                    if hasattr(Messages, key):
                        field = getattr(Messages, key)
                        if isinstance(value, dict):
                            # 处理 MongoDB 风格的操作符
                            for op, op_value in value.items():
                                if op == "$gt":
                                    conditions.append(field > op_value)
                                elif op == "$lt":
                                    conditions.append(field < op_value)
                                elif op == "$gte":
                                    conditions.append(field >= op_value)
                                elif op == "$lte":
                                    conditions.append(field <= op_value)
                                elif op == "$ne":
                                    conditions.append(field != op_value)
                                elif op == "$in":
                                    conditions.append(field.in_(op_value))
                                elif op == "$nin":
                                    conditions.append(field.not_in(op_value))
                                else:
                                    logger.warning(f"过滤器中遇到未知操作符 '{op}' (字段: '{key}')。将跳过此操作符。")
                        else:
                            # 直接相等比较
                            conditions.append(field == value)
                    else:
                        logger.warning(f"过滤器键 '{key}' 在 Messages 模型中未找到。将跳过此条件。")
                if conditions:
                    query = query.where(*conditions)

            if filter_bot:
                query = query.where(Messages.user_id != str(global_config.bot.qq_account))

            if filter_command:
                query = query.where(not_(Messages.is_command))

            # 🔥 过滤无意义消息（用于表达学习等场景）
            if filter_meaningless:
                # 排除：纯表情包、通知消息、公告消息、命令消息
                query = query.where(not_(Messages.is_emoji))
                query = query.where(not_(Messages.is_notify))
                query = query.where(not_(Messages.is_public_notice))
                query = query.where(not_(Messages.is_command))

            # 统一做上限保护，防止无限制查询导致内存暴涨
            if limit <= 0:
                capped_limit = SAFE_FETCH_LIMIT
                logger.warning(
                    f"find_messages 未指定 limit，自动限制为 {capped_limit} 行以避免内存占用过高",
                )
            else:
                capped_limit = max(1, int(limit))
                if capped_limit > SAFE_FETCH_LIMIT:
                    logger.warning(
                        f"find_messages 请求的 limit={limit} 超过安全上限，已限制为 {SAFE_FETCH_LIMIT}",
                    )
                    capped_limit = SAFE_FETCH_LIMIT

            if capped_limit > 0:
                # 如果调用方原本请求无限制并且提供了排序，保留自定义排序
                requested_unbounded = limit <= 0
                custom_sorted = False
                if requested_unbounded and sort:
                    sort_terms = []
                    for field_name, direction in sort:
                        if hasattr(Messages, field_name):
                            field = getattr(Messages, field_name)
                            if direction == 1:  # ASC
                                sort_terms.append(field.asc())
                            elif direction == -1:  # DESC
                                sort_terms.append(field.desc())
                            else:
                                logger.warning(f"字段 '{field_name}' 的排序方向 '{direction}' 无效。将跳过此排序条件。")
                        else:
                            logger.warning(f"排序字段 '{field_name}' 在 Messages 模型中未找到。将跳过此排序条件。")
                    if sort_terms:
                        query = query.order_by(*sort_terms)
                        custom_sorted = True

                if not custom_sorted:
                    if limit_mode == "earliest":
                        # 获取时间最早的 limit 条记录，已经是正序
                        query = query.order_by(Messages.time.asc())
                    else:  # 默认为 'latest'
                        # 获取时间最晚的 limit 条记录
                        query = query.order_by(Messages.time.desc())

                query = query.limit(capped_limit)
                try:
                    result = await session.execute(query)
                    fetched = result.scalars().all()
                    if custom_sorted:
                        results = fetched
                    elif limit_mode == "earliest":
                        results = fetched
                    else:
                        # latest 分支需要正序返回
                        results = sorted(fetched, key=lambda msg: msg.time)
                except Exception as e:
                    logger.error(f"执行查询失败: {e}")
                    results = []
            else:
                results = []

            # 在会话内将结果转换为字典，避免会话分离错误
            return [_model_to_dict(msg) for msg in results]
    except Exception as e:
        log_message = (
            f"使用 SQLAlchemy 查找消息失败 (filter={message_filter}, sort={sort}, limit={limit}, limit_mode={limit_mode}): {e}\n"
            + traceback.format_exc()
        )
        logger.error(log_message)
        return []


async def count_messages(message_filter: dict[str, Any]) -> int:
    """
    根据提供的过滤器计算消息数量。

    Args:
        message_filter: 查询过滤器字典，键为模型字段名，值为期望值或包含操作符的字典 (例如 {'$gt': value}).

    Returns:
        符合条件的消息数量，如果出错则返回 0。
    """
    try:
        async with get_db_session() as session:
            query = select(func.count(Messages.id))

            # 应用过滤器
            if message_filter:
                conditions = []
                for key, value in message_filter.items():
                    if hasattr(Messages, key):
                        field = getattr(Messages, key)
                        if isinstance(value, dict):
                            # 处理 MongoDB 风格的操作符
                            for op, op_value in value.items():
                                if op == "$gt":
                                    conditions.append(field > op_value)
                                elif op == "$lt":
                                    conditions.append(field < op_value)
                                elif op == "$gte":
                                    conditions.append(field >= op_value)
                                elif op == "$lte":
                                    conditions.append(field <= op_value)
                                elif op == "$ne":
                                    conditions.append(field != op_value)
                                elif op == "$in":
                                    conditions.append(field.in_(op_value))
                                elif op == "$nin":
                                    conditions.append(field.not_in(op_value))
                                else:
                                    logger.warning(
                                        f"计数时，过滤器中遇到未知操作符 '{op}' (字段: '{key}')。将跳过此操作符。"
                                    )
                        else:
                            # 直接相等比较
                            conditions.append(field == value)
                    else:
                        logger.warning(f"计数时，过滤器键 '{key}' 在 Messages 模型中未找到。将跳过此条件。")
                if conditions:
                    query = query.where(*conditions)

            count = (await session.execute(query)).scalar()
            return count or 0
    except Exception as e:
        log_message = f"使用 SQLAlchemy 计数消息失败 (message_filter={message_filter}): {e}\n{traceback.format_exc()}"
        logger.error(log_message)
        return 0


async def count_and_length_messages(message_filter: dict[str, Any]) -> tuple[int, int]:
    """
    计算符合条件的消息数量以及 processed_plain_text 的总长度。

    Args:
        message_filter: 查询过滤器字典

    Returns:
        (消息数量, 文本总长度)，出错时返回 (0, 0)
    """
    try:
        async with get_db_session() as session:
            count_expr = func.count(Messages.id)
            length_expr = func.coalesce(func.sum(func.length(Messages.processed_plain_text)), 0)
            query = select(count_expr, length_expr)

            if message_filter:
                conditions = []
                for key, value in message_filter.items():
                    if hasattr(Messages, key):
                        field = getattr(Messages, key)
                        if isinstance(value, dict):
                            for op, op_value in value.items():
                                if op == "$gt":
                                    conditions.append(field > op_value)
                                elif op == "$lt":
                                    conditions.append(field < op_value)
                                elif op == "$gte":
                                    conditions.append(field >= op_value)
                                elif op == "$lte":
                                    conditions.append(field <= op_value)
                                elif op == "$ne":
                                    conditions.append(field != op_value)
                                elif op == "$in":
                                    conditions.append(field.in_(op_value))
                                elif op == "$nin":
                                    conditions.append(field.not_in(op_value))
                                else:
                                    logger.warning(
                                        f"计数长度时，过滤器中遇到未知操作符 '{op}' (字段: '{key}')。将跳过此操作符。"
                                    )
                        else:
                            conditions.append(field == value)
                    else:
                        logger.warning(f"计数长度时，过滤器键 '{key}' 在 Messages 模型中未找到。将跳过此条件。")

                if conditions:
                    query = query.where(*conditions)

            count_value, total_length = (await session.execute(query)).one()
            return int(count_value or 0), int(total_length or 0)
    except Exception as e:
        log_message = (
            f"使用 SQLAlchemy 统计消息数量与长度失败 (filter={message_filter}): {e}\n"
            + traceback.format_exc()
        )
        logger.error(log_message)
        return 0, 0


# 你可以在这里添加更多与 messages 集合相关的数据库操作函数，例如 find_one_message, insert_message 等。
# 注意：对于 SQLAlchemy，插入操作通常是使用 await session.add() 和 await session.commit()。
# 查找单个消息可以使用 session.execute(select(Messages).where(...)).scalar_one_or_none()。


async def get_user_messages_from_streams(
    user_ids: list[str],
    stream_ids: list[str],
    timestamp_after: float,
    limit_per_stream: int,
) -> dict[str, list[dict[str, Any]]]:
    """
    一次性从多个聊天流中获取特定用户的近期消息。

    Args:
        user_ids: 目标用户的ID列表。
        stream_ids: 要查询的聊天流ID列表。
        timestamp_after: 只获取此时间戳之后的消息。
        limit_per_stream: 每个聊天流中获取该用户的消息数量上限。

    Returns:
        一个字典，键为 stream_id，值为该聊天流中的消息列表。
    """
    if not stream_ids or not user_ids:
        return {}

    try:
        async with get_db_session() as session:
            # 使用 CTE 和 row_number() 来为每个聊天流中的用户消息进行排序和编号
            ranked_messages_cte = (
                select(
                    Messages,
                    func.row_number().over(partition_by=Messages.chat_id, order_by=Messages.time.desc()).label("row_num"),
                )
                .where(
                    Messages.user_id.in_(user_ids),
                    Messages.chat_id.in_(stream_ids),
                    Messages.time > timestamp_after,
                )
                .cte("ranked_messages")
            )

            # 从 CTE 中选择每个聊天流最新的 `limit_per_stream` 条消息
            query = select(ranked_messages_cte).where(ranked_messages_cte.c.row_num <= limit_per_stream)

            result = await session.execute(query)
            messages = result.all()

            # 按 stream_id 分组
            messages_by_stream = defaultdict(list)
            for row in messages:
                # Since the row is a Row object from a CTE, we need to manually construct the model instance
                msg_instance = Messages(**{c.name: getattr(row, c.name) for c in Messages.__table__.columns})
                msg_dict = _model_to_dict(msg_instance)
                messages_by_stream[msg_dict["chat_id"]].append(msg_dict)

            # 对每个流内的消息按时间升序排序
            for stream_id in messages_by_stream:
                messages_by_stream[stream_id].sort(key=lambda m: m["time"])

            return dict(messages_by_stream)

    except Exception as e:
        log_message = (
            f"使用 SQLAlchemy 批量查找用户消息失败 (user_ids={user_ids}, streams={len(stream_ids)}): {e}\n"
            + traceback.format_exc()
        )
        logger.error(log_message)
        return {}
