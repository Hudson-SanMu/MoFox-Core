"""
Kokoro Flow Chatter - Planner

规划器：负责分析情境并生成行动计划
- 输入：会话状态、用户消息、情境类型
- 输出：LLMResponse（包含 thought、actions、expected_reaction、max_wait_seconds）
- 不负责生成具体回复文本，只决定"要做什么"
"""

from typing import TYPE_CHECKING, Optional

from src.common.logger import get_logger
from src.plugin_system.apis import llm_api
from src.utils.json_parser import extract_and_parse_json

from .models import LLMResponse
from .prompt.builder import get_prompt_builder
from .session import KokoroSession

if TYPE_CHECKING:
    from src.chat.message_receive.chat_stream import ChatStream

logger = get_logger("kfc_planner")


async def generate_plan(
    session: KokoroSession,
    user_name: str,
    situation_type: str = "new_message",
    chat_stream: Optional["ChatStream"] = None,
    available_actions: Optional[dict] = None,
    extra_context: Optional[dict] = None,
) -> LLMResponse:
    """
    生成行动计划
    
    Args:
        session: 会话对象
        user_name: 用户名称
        situation_type: 情况类型
        chat_stream: 聊天流对象
        available_actions: 可用动作字典
        extra_context: 额外上下文
        
    Returns:
        LLMResponse 对象，包含计划信息
    """
    try:
        # 1. 构建规划器提示词
        prompt_builder = get_prompt_builder()
        prompt = await prompt_builder.build_planner_prompt(
            session=session,
            user_name=user_name,
            situation_type=situation_type,
            chat_stream=chat_stream,
            available_actions=available_actions,
            extra_context=extra_context,
        )
        
        from src.config.config import global_config
        if global_config and global_config.debug.show_prompt:
            logger.info(f"[KFC Planner] 生成的规划提示词:\n{prompt}")
        
        # 2. 获取 planner 模型配置并调用 LLM
        models = llm_api.get_available_models()
        planner_config = models.get("planner")
        
        if not planner_config:
            logger.error("[KFC Planner] 未找到 planner 模型配置")
            return LLMResponse.create_error_response("未找到 planner 模型配置")
        
        success, raw_response, reasoning, model_name = await llm_api.generate_with_model(
            prompt=prompt,
            model_config=planner_config,
            request_type="kokoro_flow_chatter.plan",
        )
        
        if not success:
            logger.error(f"[KFC Planner] LLM 调用失败: {raw_response}")
            return LLMResponse.create_error_response(raw_response)
        
        logger.debug(f"[KFC Planner] LLM 响应 (model={model_name}):\n{raw_response}")
        
        # 3. 解析响应
        return _parse_response(raw_response)
        
    except Exception as e:
        logger.error(f"[KFC Planner] 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return LLMResponse.create_error_response(str(e))


def _parse_response(raw_response: str) -> LLMResponse:
    """解析 LLM 响应"""
    data = extract_and_parse_json(raw_response, strict=False)
    
    if not data or not isinstance(data, dict):
        logger.warning(f"[KFC Planner] 无法解析 JSON: {raw_response[:200]}...")
        return LLMResponse.create_error_response("无法解析响应格式")
    
    response = LLMResponse.from_dict(data)
    
    if response.thought:
        # 使用 logger 输出美化日志（颜色通过 logger 系统配置）
        logger.info(f"💭 {response.thought}")
        
        actions_str = ", ".join(a.type for a in response.actions)
        logger.debug(f"actions={actions_str}")
    else:
        logger.warning("响应缺少 thought")
    
    return response
