# -*- coding: utf-8 -*-
"""
@File    :   utils_model.py
@Time    :   2024/05/24 17:15:00
@Author  :   墨墨
@Version :   2.0 (Refactored)
@Desc    :   LLM请求协调器
"""
import time
from typing import Tuple, List, Dict, Optional, Any

from src.common.logger import get_logger
from src.config.api_ada_configs import TaskConfig, ModelInfo, UsageRecord
from .llm_utils import build_tool_options, normalize_image_format
from .model_selector import ModelSelector
from .payload_content.message import MessageBuilder
from .payload_content.tool_option import ToolCall
from .prompt_processor import PromptProcessor
from .request_strategy import RequestStrategy
from .utils import llm_usage_recorder

logger = get_logger("model_utils")

class LLMRequest:
    """
    LLM请求协调器。
    封装了模型选择、Prompt处理、请求执行和高级策略（如故障转移、并发）的完整流程。
    为上层业务逻辑提供统一的、简化的接口来与大语言模型交互。
    """

    def __init__(self, model_set: TaskConfig, request_type: str = "") -> None:
        """
        初始化LLM请求协调器。

        Args:
            model_set (TaskConfig): 特定任务的模型配置集合。
            request_type (str, optional): 请求类型或任务名称，用于日志和用量记录。 Defaults to "".
        """
        self.task_name = request_type
        self.model_for_task = model_set
        self.request_type = request_type
        self.model_selector = ModelSelector(model_set, request_type)
        self.prompt_processor = PromptProcessor()
        self.request_strategy = RequestStrategy(model_set, self.model_selector, request_type)

    async def generate_response_for_image(
        self,
        prompt: str,
        image_base64: str,
        image_format: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Tuple[str, str, Optional[List[ToolCall]]]]:
        """
        为包含图像的多模态输入生成文本响应。

        Args:
            prompt (str): 文本提示。
            image_base64 (str): Base64编码的图像数据。
            image_format (str): 图像格式 (例如, "png", "jpeg")。
            temperature (Optional[float], optional): 控制生成文本的随机性。 Defaults to None.
            max_tokens (Optional[int], optional): 生成响应的最大长度。 Defaults to None.

        Returns:
            Tuple[str, Tuple[str, str, Optional[List[ToolCall]]]]:
                - 清理后的响应内容。
                - 一个元组，包含思考过程、模型名称和工具调用列表。
        """
        start_time = time.time()
        
        # 步骤 1: 选择一个支持图像处理的模型
        model_info, api_provider, client = self.model_selector.select_model()
        
        # 步骤 2: 准备消息体
        # 预处理文本提示
        processed_prompt = self.prompt_processor.process_prompt(prompt, model_info, api_provider, self.task_name)
        # 规范化图像格式
        normalized_format = normalize_image_format(image_format)
        
        # 使用MessageBuilder构建多模态消息
        message_builder = MessageBuilder()
        message_builder.add_text_content(processed_prompt)
        message_builder.add_image_content(
            image_base64=image_base64,
            image_format=normalized_format,
            support_formats=client.get_support_image_formats(),
        )
        messages = [message_builder.build()]

        # 步骤 3: 执行请求 (图像请求通常不走复杂的故障转移策略，直接执行)
        from .request_executor import RequestExecutor
        executor = RequestExecutor(
            task_name=self.task_name,
            model_set=self.model_for_task,
            api_provider=api_provider,
            client=client,
            model_info=model_info,
            model_selector=self.model_selector,
        )
        response = await executor.execute_request(
            request_type="response",
            message_list=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # 步骤 4: 处理响应
        content, reasoning_content = self.prompt_processor.extract_reasoning(response.content or "")
        tool_calls = response.tool_calls
        
        # 记录用量
        if usage := response.usage:
            await self._record_usage(model_info, usage, time.time() - start_time)
            
        return content, (reasoning_content, model_info.name, tool_calls)

    async def generate_response_for_voice(self, voice_base64: str) -> Optional[str]:
        """
        将语音数据转换为文本（语音识别）。

        Args:
            voice_base64 (str): Base64编码的语音数据。

        Returns:
            Optional[str]: 识别出的文本内容，如果失败则返回None。
        """
        # 选择一个支持语音识别的模型
        model_info, api_provider, client = self.model_selector.select_model()
        
        from .request_executor import RequestExecutor
        # 创建请求执行器
        executor = RequestExecutor(
            task_name=self.task_name,
            model_set=self.model_for_task,
            api_provider=api_provider,
            client=client,
            model_info=model_info,
            model_selector=self.model_selector,
        )
        # 执行语音转文本请求
        response = await executor.execute_request(
            request_type="audio",
            audio_base64=voice_base64,
        )
        return response.content or None

    async def generate_response_async(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        raise_when_empty: bool = True,
    ) -> Tuple[str, Tuple[str, str, Optional[List[ToolCall]]]]:
        """
        异步生成文本响应，支持并发和故障转移等高级策略。

        Args:
            prompt (str): 用户输入的提示。
            temperature (Optional[float], optional): 控制生成文本的随机性。 Defaults to None.
            max_tokens (Optional[int], optional): 生成响应的最大长度。 Defaults to None.
            tools (Optional[List[Dict[str, Any]]], optional): 可供模型调用的工具列表。 Defaults to None.
            raise_when_empty (bool, optional): 如果最终响应为空，是否抛出异常。 Defaults to True.

        Returns:
            Tuple[str, Tuple[str, str, Optional[List[ToolCall]]]]:
                - 清理后的响应内容。
                - 一个元组，包含思考过程、最终使用的模型名称和工具调用列表。
        """
        start_time = time.time()
        
        # 步骤 1: 准备基础请求载荷
        tool_built = build_tool_options(tools)
        base_payload = {
            "prompt": prompt,
            "tool_options": tool_built,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_processor": self.prompt_processor,
        }
        
        # 步骤 2: 根据配置选择执行策略 (并发或单次带故障转移)
        concurrency_count = getattr(self.model_for_task, "concurrency_count", 1)
        
        if concurrency_count <= 1:
            # 单次请求，但使用带故障转移的策略
            result = await self.request_strategy.execute_with_fallback(
                base_payload, raise_when_empty
            )
        else:
            # 并发请求策略
            result = await self.request_strategy.execute_concurrently(
                self.request_strategy.execute_with_fallback,
                concurrency_count,
                base_payload,
                raise_when_empty=False, # 在并发模式下，单个任务失败不应立即抛出异常
            )
        
        # 步骤 3: 处理最终结果
        content = result.get("content", "")
        reasoning_content = result.get("reasoning_content", "")
        model_name = result.get("model_name", "unknown")
        tool_calls = result.get("tool_calls")
        
        # 步骤 4: 记录用量 (从策略返回的结果中获取最终使用的模型信息和用量)
        final_model_info = result.get("model_info")
        usage = result.get("usage")

        if final_model_info and usage:
            await self._record_usage(final_model_info, usage, time.time() - start_time)
        
        return content, (reasoning_content, model_name, tool_calls)

    async def get_embedding(self, embedding_input: str) -> Tuple[List[float], str]:
        """
        获取给定文本的嵌入向量 (Embedding)。

        Args:
            embedding_input (str): 需要进行嵌入的文本。

        Returns:
            Tuple[List[float], str]: 嵌入向量列表和所使用的模型名称。
        
        Raises:
            RuntimeError: 如果获取embedding失败。
        """
        start_time = time.time()
        # 选择一个支持embedding的模型
        model_info, api_provider, client = self.model_selector.select_model()
        
        from .request_executor import RequestExecutor
        # 创建请求执行器
        executor = RequestExecutor(
            task_name=self.task_name,
            model_set=self.model_for_task,
            api_provider=api_provider,
            client=client,
            model_info=model_info,
            model_selector=self.model_selector,
        )
        # 执行embedding请求
        response = await executor.execute_request(
            request_type="embedding",
            embedding_input=embedding_input,
        )
        
        embedding = response.embedding
        if not embedding:
            raise RuntimeError("获取embedding失败")
            
        # 记录用量
        if usage := response.usage:
            await self._record_usage(model_info, usage, time.time() - start_time, "/embeddings")
            
        return embedding, model_info.name

    async def _record_usage(self, model_info: ModelInfo, usage: UsageRecord, time_cost: float, endpoint: str = "/chat/completions"):
        """
        记录模型API的调用用量到数据库。

        Args:
            model_info (ModelInfo): 使用的模型信息。
            usage (UsageRecord): 包含token用量信息的对象。
            time_cost (float): 本次请求的总耗时（秒）。
            endpoint (str, optional): 请求的API端点。 Defaults to "/chat/completions".
        """
        await llm_usage_recorder.record_usage_to_database(
            model_info=model_info,
            model_usage=usage,
            user_id="system",  # 当前所有请求都以系统用户身份记录
            time_cost=time_cost,
            request_type=self.request_type,
            endpoint=endpoint,
        )
