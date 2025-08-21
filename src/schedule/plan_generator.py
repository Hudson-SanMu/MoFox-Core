# mmc/src/schedule/plan_generator.py

import json
from typing import List
from pydantic import BaseModel, ValidationError
from json_repair import repair_json

from src.config.config import global_config, model_config
from src.llm_models.utils_model import LLMRequest
from src.common.logger import get_logger

logger = get_logger("plan_generator")

class PlanResponse(BaseModel):
    """
    用于验证月度计划LLM响应的Pydantic模型。
    """
    plans: List[str]

class PlanGenerator:
    """
    负责生成月度计划。
    """

    def __init__(self):
        self.bot_personality = self._get_bot_personality()
        task_config = model_config.model_task_config.get_task("monthly_plan_generator")
        self.llm_request = LLMRequest(model_set=task_config, request_type="monthly_plan_generator")

    def _get_bot_personality(self) -> str:
        """
        从全局配置中获取Bot的人设描述。
        """
        core = global_config.personality.personality_core or ""
        side = global_config.personality.personality_side or ""
        identity = global_config.personality.identity or ""
        return f"核心人设: {core}\n侧面人设: {side}\n身份设定: {identity}"

    def _build_prompt(self, year: int, month: int, count: int) -> str:
        """
        构建用于生成月度计划的Prompt。
        """
        prompt_template = f"""
        你是一个富有想象力的助手，你的任务是为一位虚拟角色生成月度计划。
        
        **角色设定:**
        ---
        {self.bot_personality}
        ---

        请为即将到来的 **{year}年{month}月** 设计 **{count}** 个符合该角色身份的、独立的、积极向上的月度计划或小目标。

        **要求:**
        1.  每个计划都应简短、清晰，用一两句话描述。
        2.  语言风格必须自然、口语化，严格符合角色的性格设定。
        3.  计划内容要具有创造性，避免陈词滥调。
        4.  请以严格的JSON格式返回，格式为：{{"plans": ["计划一", "计划二", ...]}}
        5.  除了JSON对象，不要包含任何额外的解释、注释或前后导语。
        """
        return prompt_template.strip()

    async def generate_plans(self, year: int, month: int, count: int) -> List[str]:
        """
        调用LLM生成指定月份的计划。

        :param year: 年份
        :param month: 月份
        :param count: 需要生成的计划数量
        :return: 生成的计划文本列表
        """
        try:
            # 1. 构建Prompt
            prompt = self._build_prompt(year, month, count)
            logger.info(f"正在为 {year}-{month} 生成 {count} 个月度计划...")

            # 2. 调用LLM
            llm_content, (reasoning, model_name, _) = await self.llm_request.generate_response_async(prompt=prompt)
            
            logger.info(f"使用模型 '{model_name}' 生成完成。")
            if reasoning:
                logger.debug(f"模型推理过程: {reasoning}")

            if not llm_content:
                logger.error("LLM未能返回有效的计划内容。")
                return []

            # 3. 解析并验证LLM返回的JSON
            try:
                # 移除可能的Markdown代码块标记
                clean_content = llm_content.strip()
                if clean_content.startswith("```json"):
                    clean_content = clean_content[7:]
                if clean_content.endswith("```"):
                    clean_content = clean_content[:-3]
                
                # 修复并解析JSON
                repaired_json_str = repair_json(clean_content)
                data = json.loads(repaired_json_str)

                # 使用Pydantic进行验证
                validated_response = PlanResponse.model_validate(data)
                plans = validated_response.plans
                
                logger.info(f"成功生成并验证了 {len(plans)} 个月度计划。")
                return plans

            except json.JSONDecodeError:
                logger.error(f"修复后仍然无法解析LLM返回的JSON: {llm_content}")
                return []
            except ValidationError as e:
                logger.error(f"LLM返回的JSON格式不符合预期: {e}\n原始响应: {llm_content}")
                return []

        except Exception as e:
            logger.error(f"调用LLM生成月度计划时发生未知错误: {e}", exc_info=True)
            return []