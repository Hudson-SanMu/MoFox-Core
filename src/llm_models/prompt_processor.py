# -*- coding: utf-8 -*-
"""
@File    :   prompt_processor.py
@Time    :   2024/05/24 16:45:00
@Author  :   墨墨
@Version :   1.0
@Desc    :   提示词预处理（反截断、内容混淆等）
"""
import random
import string
import re
from typing import Tuple

from src.common.logger import get_logger
from src.config.api_ada_configs import APIProvider, ModelInfo

logger = get_logger("prompt_processor")


class PromptProcessor:
    """
    提示词处理器。
    负责对发送给模型的原始prompt进行预处理，以增强模型性能或实现特定功能。
    主要功能包括：
    1.  **反截断**：在prompt末尾添加一个特殊的结束标记指令，帮助判断模型输出是否被截断。
    2.  **内容混淆**：向prompt中注入随机的“噪音”字符串，并附带指令让模型忽略它们，
        可能用于绕过某些平台的审查或内容策略。
    3.  **思维链提取**：从模型的响应中分离出思考过程（被<think>标签包裹）和最终答案。
    """

    def __init__(self):
        """初始化Prompt处理器，定义所需的指令文本。"""
        # 指导模型忽略噪音字符串的指令
        self.noise_instruction = """**【核心过滤规则】**
在交互过程中,你会在用户的输入内容中看到一些无意义的、由字母和数字组成的随机字符串。这些字符串是API传输过程中的技术伪装信息,【不是】用户真实意图的一部分。

你的任务是【完全并彻底地忽略】这些随机字符串。
**【严禁】** 将这些字符串写入你的回复、对它们进行任何形式的解读、或将它们作为故事的一部分进行模拟。当看到这些字符串时,请将其视为空白或直接跳过,并根据用户其他有意义的输入(如有)或当前故事情节来继续你的叙事。此规则为最高优先级。
"""
        # 定义一个独特的结束标记，用于反截断检查
        self.end_marker = "###MAI_RESPONSE_END###"
        # 指导模型在回复末尾添加结束标记的指令
        self.anti_truncation_instruction = f"""
**【输出完成信令】**
这是一个非常重要的指令,请务-务必遵守。在你的回复内容完全结束后,请务必在最后另起一行,只写 `{self.end_marker}` 作为结束标志。
例如:
<你的回复内容>
{self.end_marker}

这有助于我判断你的输出是否被截断。请不要在 `{self.end_marker}` 前后添加任何其他文字或标点。
"""

    def process_prompt(
        self, prompt: str, model_info: ModelInfo, api_provider: APIProvider, task_name: str
    ) -> str:
        """
        根据模型和API提供商的配置，对输入的prompt进行预处理。

        Args:
            prompt (str): 原始的用户输入prompt。
            model_info (ModelInfo): 当前使用的模型信息。
            api_provider (APIProvider): 当前API提供商的配置。
            task_name (str): 当前任务的名称，用于日志记录。

        Returns:
            str: 经过处理后的、最终将发送给模型的prompt。
        """
        processed_prompt = prompt

        # 步骤 1: 根据模型配置添加反截断指令
        use_anti_truncation = getattr(model_info, "use_anti_truncation", False)
        if use_anti_truncation:
            processed_prompt += self.anti_truncation_instruction
            logger.info(f"模型 '{model_info.name}' (任务: '{task_name}') 已启用反截断功能。")

        # 步骤 2: 根据API提供商配置应用内容混淆
        if getattr(api_provider, "enable_content_obfuscation", False):
            intensity = getattr(api_provider, "obfuscation_intensity", 1)
            logger.info(f"为API提供商 '{api_provider.name}' 启用内容混淆，强度级别: {intensity}")
            processed_prompt = self._apply_content_obfuscation(processed_prompt, intensity)

        return processed_prompt

    def _apply_content_obfuscation(self, text: str, intensity: int) -> str:
        """
        对文本应用内容混淆处理。
        首先添加过滤规则指令，然后注入随机噪音。
        """
        # 在文本开头加入指导模型忽略噪音的指令
        processed_text = self.noise_instruction + "\n\n" + text
        logger.debug(f"已添加过滤规则指令，文本长度: {len(text)} -> {len(processed_text)}")

        # 在文本中注入随机乱码
        final_text = self._inject_random_noise(processed_text, intensity)
        logger.debug(f"乱码注入完成，最终文本长度: {len(final_text)}")

        return final_text

    @staticmethod
    def _inject_random_noise(text: str, intensity: int) -> str:
        """
        根据指定的强度，在文本的词语之间随机注入噪音字符串。

        Args:
            text (str): 待注入噪音的文本。
            intensity (int): 混淆强度 (1, 2, or 3)，决定噪音的注入概率和长度。

        Returns:
            str: 注入噪音后的文本。
        """
        def generate_noise(length: int) -> str:
            """生成指定长度的随机噪音字符串。"""
            chars = (
                string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?"
                + "一二三四五六七八九零壹贰叁" + "αβγδεζηθικλμνξοπρστυφχψω" + "∀∃∈∉∪∩⊂⊃∧∨¬→↔∴∵"
            )
            return "".join(random.choice(chars) for _ in range(length))

        # 根据强度级别定义注入参数
        params = {
            1: {"probability": 15, "length": (3, 6)},   # 低强度
            2: {"probability": 25, "length": (5, 10)},  # 中强度
            3: {"probability": 35, "length": (8, 15)},  # 高强度
        }
        config = params.get(intensity, params[1]) # 默认为低强度
        logger.debug(f"乱码注入参数: 概率={config['probability']}%, 长度范围={config['length']}")

        words = text.split()
        result = []
        noise_count = 0
        for word in words:
            result.append(word)
            # 按概率决定是否注入噪音
            if random.randint(1, 100) <= config["probability"]:
                noise_length = random.randint(*config["length"])
                noise = generate_noise(noise_length)
                result.append(noise)
                noise_count += 1

        logger.debug(f"共注入 {noise_count} 个乱码片段，原词数: {len(words)}")
        return " ".join(result)
    
    @staticmethod
    def extract_reasoning(content: str) -> Tuple[str, str]:
        """
        从模型返回的完整内容中提取被<think>...</think>标签包裹的思考过程，
        并返回清理后的内容和思考过程。

        Args:
            content (str): 模型返回的原始字符串。

        Returns:
            Tuple[str, str]:
                - 清理后的内容（移除了<think>标签及其内容）。
                - 提取出的思考过程文本（如果没有则为空字符串）。
        """
        # 使用正则表达式查找<think>标签
        match = re.search(r"(?:<think>)?(.*?)</think>", content, re.DOTALL)
        # 从内容中移除<think>标签及其包裹的所有内容
        clean_content = re.sub(r"(?:<think>)?.*?</think>", "", content, flags=re.DOTALL, count=1).strip()
        # 如果找到匹配项，则提取思考过程
        reasoning = match.group(1).strip() if match else ""
        return clean_content, reasoning
