import os
import traceback

from mofox_bus.runtime import MessageRuntime
from mofox_bus import MessageEnvelope
from src.chat.message_manager import message_manager
from src.common.logger import get_logger
from src.config.config import global_config
from src.chat.message_receive.chat_stream import ChatStream, get_chat_manager
from src.common.data_models.database_data_model import DatabaseGroupInfo, DatabaseUserInfo, DatabaseMessages

runtime = MessageRuntime()

# 获取项目根目录（假设本文件在src/chat/message_receive/下，根目录为上上上级目录）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# 配置主程序日志格式
logger = get_logger("chat")

class MessageHandler:
    def __init__(self):
        self._started = False

    async def preprocess(self, chat: ChatStream, message: DatabaseMessages):
        # message 已经是 DatabaseMessages，直接使用
        group_info = chat.group_info

        # 先交给消息管理器处理
        try:
            # 在将消息添加到管理器之前进行最终的静默检查
            should_process_in_manager = True
            if group_info and str(group_info.group_id) in global_config.message_receive.mute_group_list:
                # 检查消息是否为图片或表情包
                is_image_or_emoji = message.is_picid or message.is_emoji
                if not message.is_mentioned and not is_image_or_emoji:
                    logger.debug(f"群组 {group_info.group_id} 在静默列表中，且消息不是@、回复或图片/表情包，跳过消息管理器处理")
                    should_process_in_manager = False
                elif is_image_or_emoji:
                    logger.debug(f"群组 {group_info.group_id} 在静默列表中，但消息是图片/表情包，静默处理")
                    should_process_in_manager = False

            if should_process_in_manager:
                await message_manager.add_message(chat.stream_id, message)
                logger.debug(f"消息已添加到消息管理器: {chat.stream_id}")

        except Exception as e:
            logger.error(f"消息添加到消息管理器失败: {e}")

        # 存储消息到数据库，只进行一次写入
        try:
            await MessageStorage.store_message(message, chat)
        except Exception as e:
            logger.error(f"存储消息到数据库失败: {e}")
            traceback.print_exc()

        # 情绪系统更新 - 在消息存储后触发情绪更新
        try:
            if global_config.mood.enable_mood:
                # 获取兴趣度用于情绪更新
                interest_rate = message.interest_value
                if interest_rate is None:
                    interest_rate = 0.0
                logger.debug(f"开始更新情绪状态，兴趣度: {interest_rate:.2f}")

                # 获取当前聊天的情绪对象并更新情绪状态
                chat_mood = mood_manager.get_mood_by_chat_id(chat.stream_id)
                await chat_mood.update_mood_by_message(message, interest_rate)
                logger.debug("情绪状态更新完成")
        except Exception as e:
            logger.error(f"更新情绪状态失败: {e}")
            traceback.print_exc()


    async def handle_message(self, envelope: MessageEnvelope):
        # 控制握手等消息可能缺少 message_info，这里直接跳过避免 KeyError
        message_info = envelope.get("message_info")
        if not isinstance(message_info, dict):
            logger.debug(
                "收到缺少 message_info 的消息，已跳过。可用字段: %s",
                ", ".join(envelope.keys()),
            )
            return

        if message_info.get("group_info") is not None:
            message_info["group_info"]["group_id"] = str( # type: ignore
                message_info["group_info"]["group_id"] # type: ignore
            ) 
        if message_info.get("user_info") is not None:
            message_info["user_info"]["user_id"] = str( # type: ignore
                message_info["user_info"]["user_id"] # type: ignore
            )

        group_info = message_info.get("group_info")
        user_info = message_info.get("user_info")

        chat_stream = await get_chat_manager().get_or_create_stream(
            platform=envelope["platform"],  # type: ignore
            user_info=user_info,  # type: ignore
            group_info=group_info,
        )

        # 生成 DatabaseMessages
        from src.chat.message_receive.message_processor import process_message_from_dict
        message = await process_message_from_dict(
            message_dict=envelope,
            stream_id=chat_stream.stream_id,
            platform=chat_stream.platform
        )

        