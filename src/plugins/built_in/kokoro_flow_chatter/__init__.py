"""
Kokoro Flow Chatter (KFC) - 私聊特化的心流聊天器

重构版本，支持双模式架构：

工作模式：
- unified（统一模式）: 单次 LLM 调用完成思考和回复生成（默认）
- split（分离模式）: Planner + Replyer 两次 LLM 调用，更精细的控制

核心设计理念：
1. Chatter 职责极简化：只负责"收到消息 → 规划执行"
2. Session 状态简化：只有 IDLE 和 WAITING 两种状态
3. 独立的 Replyer：专属的提示词构建和 LLM 交互
4. 独立的主动思考器：负责等待管理和主动发起
5. 大模板 + 小模板：线性叙事风格的提示词架构
"""

from .models import (
    EventType,
    SessionStatus,
    MentalLogEntry,
    WaitingConfig,
    ActionModel,
    LLMResponse,
)
from .session import KokoroSession, SessionManager, get_session_manager
from .chatter import KokoroFlowChatter
from .planner import generate_plan
from .replyer import generate_reply_text
from .unified import generate_unified_response
from .proactive_thinker import (
    ProactiveThinker,
    get_proactive_thinker,
    start_proactive_thinker,
    stop_proactive_thinker,
)
from .config import (
    KFCMode,
    KokoroFlowChatterConfig,
    get_config,
    load_config,
    reload_config,
)
from .plugin import KokoroFlowChatterPlugin
from src.plugin_system.base.plugin_metadata import PluginMetadata

__plugin_meta__ = PluginMetadata(
    name="Kokoro Flow Chatter",
    description="专为私聊设计的深度情感交互处理器，支持统一/分离双模式",
    usage="在私聊场景中自动启用，可通过 [kokoro_flow_chatter].enable 和 .mode 配置",
    version="3.1.0",
    author="MoFox",
    keywords=["chatter", "kokoro", "private", "emotional", "narrative", "dual-mode"],
    categories=["Chat", "AI", "Emotional"],
    extra={"is_built_in": True, "chat_type": "private"},
)

__all__ = [
    # Models
    "EventType",
    "SessionStatus",
    "MentalLogEntry",
    "WaitingConfig",
    "ActionModel",
    "LLMResponse",
    # Session
    "KokoroSession",
    "SessionManager",
    "get_session_manager",
    # Core Components
    "KokoroFlowChatter",
    "generate_plan",
    "generate_reply_text",
    "generate_unified_response",
    # Proactive Thinker
    "ProactiveThinker",
    "get_proactive_thinker",
    "start_proactive_thinker",
    "stop_proactive_thinker",
    # Config
    "KFCMode",
    "KokoroFlowChatterConfig",
    "get_config",
    "load_config",
    "reload_config",
    # Plugin
    "KokoroFlowChatterPlugin",
    "__plugin_meta__",
]
