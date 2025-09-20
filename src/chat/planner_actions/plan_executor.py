"""
PlanExecutor: 接收 Plan 对象并执行其中的所有动作。
"""
from src.chat.planner_actions.action_manager import ActionManager
from src.common.data_models.info_data_model import Plan
from src.common.logger import get_logger

logger = get_logger("plan_executor")


class PlanExecutor:
    """
    负责接收一个 Plan 对象，并执行其中最终确定的所有动作。

    这个类是规划流程的最后一步，将规划结果转化为实际的动作执行。

    Attributes:
        action_manager (ActionManager): 用于实际执行各种动作的管理器实例。
    """

    def __init__(self, action_manager: ActionManager):
        """
        初始化 PlanExecutor。

        Args:
            action_manager (ActionManager): 一个 ActionManager 实例，用于执行动作。
        """
        self.action_manager = action_manager

    @staticmethod
    async def execute(plan: Plan):
        """
        遍历并执行 Plan 对象中 `decided_actions` 列表里的所有动作。

        如果动作类型为 "no_action"，则会记录原因并跳过。
        否则，它将调用 ActionManager 来执行相应的动作。

        Args:
            plan (Plan): 包含待执行动作列表的 Plan 对象。
        """
        if not plan.decided_actions:
            logger.info("没有需要执行的动作。")
            return

        for action_info in plan.decided_actions:
            if action_info.action_type == "no_action":
                logger.info(f"规划器决策不执行动作，原因: {action_info.reasoning}")
                continue

            # TODO: 对接 ActionManager 的执行方法
            # 这是一个示例调用，需要根据 ActionManager 的最终实现进行调整
            logger.info(f"执行动作: {action_info.action_type}, 原因: {action_info.reasoning}")
            # await self.action_manager.execute_action(
            #     action_name=action_info.action_type,
            #     action_data=action_info.action_data,
            #     reasoning=action_info.reasoning,
            #     action_message=action_info.action_message,
            # )
