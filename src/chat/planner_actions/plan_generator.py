"""
PlanGenerator: 负责搜集和汇总所有决策所需的信息，生成一个未经筛选的“原始计划” (Plan)。
"""
import time
from typing import Dict, Optional, Tuple

from src.chat.utils.chat_message_builder import get_raw_msg_before_timestamp_with_chat
from src.chat.utils.utils import get_chat_type_and_target_info
from src.common.data_models.database_data_model import DatabaseMessages
from src.common.data_models.info_data_model import Plan, TargetPersonInfo
from src.config.config import global_config
from src.plugin_system.base.component_types import ActionInfo, ChatMode, ComponentType
from src.plugin_system.core.component_registry import component_registry


class PlanGenerator:
    """
    PlanGenerator 负责在规划流程的初始阶段收集所有必要信息。

    它会汇总以下信息来构建一个“原始”的 Plan 对象，该对象后续会由 PlanFilter 进行筛选：
    -   当前聊天信息 (ID, 目标用户)
    -   当前可用的动作列表
    -   最近的聊天历史记录

    Attributes:
        chat_id (str): 当前聊天的唯一标识符。
        action_manager (ActionManager): 用于获取可用动作列表的管理器。
    """

    def __init__(self, chat_id: str):
        """
        初始化 PlanGenerator。

        Args:
            chat_id (str): 当前聊天的 ID。
        """
        from src.chat.planner_actions.action_manager import ActionManager
        self.chat_id = chat_id
        # 注意：ActionManager 可能需要根据实际情况初始化
        self.action_manager = ActionManager()

    async def generate(self, mode: ChatMode) -> Plan:
        """
        收集所有信息，生成并返回一个初始的 Plan 对象。

        这个 Plan 对象包含了决策所需的所有上下文信息。

        Args:
            mode (ChatMode): 当前的聊天模式。

        Returns:
            Plan: 一个填充了初始上下文信息的 Plan 对象。
        """
        _is_group_chat, chat_target_info_dict = get_chat_type_and_target_info(self.chat_id)
        
        target_info = None
        if chat_target_info_dict:
            target_info = TargetPersonInfo(**chat_target_info_dict)

        available_actions = self._get_available_actions()
        chat_history_raw = get_raw_msg_before_timestamp_with_chat(
            chat_id=self.chat_id,
            timestamp=time.time(),
            limit=int(global_config.chat.max_context_size),
        )
        chat_history = [DatabaseMessages(**msg) for msg in chat_history_raw]


        plan = Plan(
            chat_id=self.chat_id,
            mode=mode,
            available_actions=available_actions,
            chat_history=chat_history,
            target_info=target_info,
        )
        return plan

    def _get_available_actions(self) -> Dict[str, "ActionInfo"]:
        """
        从 ActionManager 和组件注册表中获取当前所有可用的动作。

        它会合并已注册的动作和系统级动作（如 "no_reply"），
        并以字典形式返回。

        Returns:
            Dict[str, "ActionInfo"]: 一个字典，键是动作名称，值是 ActionInfo 对象。
        """
        current_available_actions_dict = self.action_manager.get_using_actions()
        all_registered_actions: Dict[str, ActionInfo] = component_registry.get_components_by_type( # type: ignore
            ComponentType.ACTION
        )
        
        current_available_actions = {}
        for action_name in current_available_actions_dict:
            if action_name in all_registered_actions:
                current_available_actions[action_name] = all_registered_actions[action_name]

        no_reply_info = ActionInfo(
            name="no_reply",
            component_type=ComponentType.ACTION,
            description="系统级动作：选择不回复消息的决策",
            action_parameters={},
            activation_keywords=[],
            plugin_name="SYSTEM",
            enabled=True,
            parallel_action=False,
        )
        current_available_actions["no_reply"] = no_reply_info
        
        return current_available_actions