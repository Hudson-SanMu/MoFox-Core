import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from src.config.config import global_config
from src.plugin_system.apis import message_api

router = APIRouter()

@router.get("/messages/recent")
async def get_message_stats(
    days: int = Query(1, ge=1, description="指定查询过去多少天的数据"),
    message_type: Literal["all", "sent", "received"] = Query("all", description="筛选消息类型: 'sent' (BOT发送的), 'received' (BOT接收的), or 'all' (全部)")
):
    """
    获取BOT在指定天数内的消息统计数据。
    """
    try:
        end_time = time.time()
        start_time = end_time - (days * 24 * 3600)

        messages = await message_api.get_messages_by_time(start_time, end_time)

        sent_count = 0
        received_count = 0
        bot_qq = str(global_config.bot.qq_account)

        for msg in messages:
            if msg.get("user_id") == bot_qq:
                sent_count += 1
            else:
                received_count += 1
        if message_type == "sent":
            return {"days": days, "message_type": message_type, "count": sent_count}
        elif message_type == "received":
            return {"days": days, "message_type": message_type, "count": received_count}
        else:
            return {
                "days": days,
                "message_type": message_type,
                "sent_count": sent_count,
                "received_count": received_count,
                "total_count": len(messages)
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/messages/stats_by_chat")
async def get_message_stats_by_chat(
    days: int = Query(1, ge=1, description="指定查询过去多少天的数据"),
    group_by_user: bool = Query(False, description="是否按用户进行分组统计"),
    filter_bot: bool = Query(False, description="是否过滤BOT自身的消息")
):
    """
    获取BOT在指定天数内按聊天流或按用户统计的消息数据。
    """
    try:
        end_time = time.time()
        start_time = end_time - (days * 24 * 3600)
        messages = await message_api.get_messages_by_time(start_time, end_time)
        bot_qq = str(global_config.bot.qq_account)

        if filter_bot:
            messages = [msg for msg in messages if msg.get("user_id") != bot_qq]

        stats = {}

        for msg in messages:
            chat_id = msg.get("chat_id", "unknown")
            user_id = msg.get("user_id")

            if chat_id not in stats:
                stats[chat_id] = {
                    "total_stats": {"sent": 0, "received": 0, "total": 0},
                    "user_stats": {}
                }

            stats[chat_id]["total_stats"]["total"] += 1
            if user_id == bot_qq:
                stats[chat_id]["total_stats"]["sent"] += 1
            else:
                stats[chat_id]["total_stats"]["received"] += 1

            if group_by_user:
                if user_id not in stats[chat_id]["user_stats"]:
                    stats[chat_id]["user_stats"][user_id] = 0
                
                stats[chat_id]["user_stats"][user_id] += 1

        if not group_by_user:
            # 如果不按用户分组，则只返回总统计信息
            return {chat_id: data["total_stats"] for chat_id, data in stats.items()}

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
