"""
Kokoro Flow Chatter - Replyer

纯粹的回复生成器：
- 接收 planner 的决策（thought 等）
- 专门负责将回复意图转化为自然的对话文本
- 不输出 JSON，直接生成可发送的消息文本
"""

from typing import TYPE_CHECKING, Optional

from src.common.logger import get_logger
from src.plugin_system.apis import llm_api

from .prompt.builder import get_prompt_builder
from .session import KokoroSession

if TYPE_CHECKING:
    from src.chat.message_receive.chat_stream import ChatStream

logger = get_logger("kfc_replyer")


async def generate_reply_text(
    session: KokoroSession,
    user_name: str,
    thought: str,
    situation_type: str = "new_message",
    chat_stream: Optional["ChatStream"] = None,
    extra_context: Optional[dict] = None,
) -> tuple[bool, str]:
    """
    生成回复文本
    
    Args:
        session: 会话对象
        user_name: 用户名称
        thought: 规划器生成的想法（内心独白）
        situation_type: 情况类型
        chat_stream: 聊天流对象
        extra_context: 额外上下文
        
    Returns:
        (success, reply_text) 元组
        - success: 是否成功生成
        - reply_text: 生成的回复文本
    """
    try:
        # 1. 构建回复器提示词
        prompt_builder = get_prompt_builder()
        prompt = await prompt_builder.build_replyer_prompt(
            session=session,
            user_name=user_name,
            thought=thought,
            situation_type=situation_type,
            chat_stream=chat_stream,
            extra_context=extra_context,
        )
        
        from src.config.config import global_config
        if global_config and global_config.debug.show_prompt:
            logger.info(f"[KFC Replyer] 生成的回复提示词:\n{prompt}")
        
        # 2. 获取 replyer 模型配置并调用 LLM
        models = llm_api.get_available_models()
        replyer_config = models.get("replyer")
        
        if not replyer_config:
            logger.error("[KFC Replyer] 未找到 replyer 模型配置")
            return False, "（回复生成失败：未找到模型配置）"
        
        success, raw_response, reasoning, model_name = await llm_api.generate_with_model(
            prompt=prompt,
            model_config=replyer_config,
            request_type="kokoro_flow_chatter.reply",
        )
        
        if not success:
            logger.error(f"[KFC Replyer] LLM 调用失败: {raw_response}")
            return False, "（回复生成失败）"
        
        # 3. 清理并返回回复文本
        reply_text = _clean_reply_text(raw_response)
        
        # 使用 logger 输出美化日志（颜色通过 logger 系统配置）
        logger.info(f"💬 {reply_text}")
        
        return True, reply_text
        
    except Exception as e:
        logger.error(f"[KFC Replyer] 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False, "（回复生成失败）"


def _clean_reply_text(raw_text: str) -> str:
    """
    清理回复文本
    
    移除可能的前后缀、引号、markdown 标记等
    """
    text = raw_text.strip()
    
    # 移除可能的 markdown 代码块标记
    if text.startswith("```") and text.endswith("```"):
        lines = text.split("\n")
        if len(lines) >= 3:
            # 移除首尾的 ``` 行
            text = "\n".join(lines[1:-1]).strip()
    
    # 移除首尾的引号（如果整个文本被引号包裹）
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    
    # 移除可能的"你说："、"回复："等前缀
    prefixes_to_remove = ["你说：", "你说:", "回复：", "回复:", "我说：", "我说:"]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    
    return text
