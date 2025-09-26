# -*- coding: utf-8 -*-
"""
@File    :   request_executor.py
@Time    :   2024/05/24 16:15:00
@Author  :   墨墨
@Version :   1.0
@Desc    :   负责执行LLM请求、处理重试及异常
"""
import asyncio
from typing import List, Callable, Optional, Tuple

from src.common.logger import get_logger
from src.config.api_ada_configs import APIProvider, ModelInfo, TaskConfig
from .exceptions import (
    NetworkConnectionError,
    ReqAbortException,
    RespNotOkException,
    RespParseException,
)
from .model_client.base_client import APIResponse, BaseClient
from .model_selector import ModelSelector
from .payload_content.message import Message
from .payload_content.resp_format import RespFormat
from .payload_content.tool_option import ToolOption
from .utils import compress_messages

logger = get_logger("model_utils")


class RequestExecutor:
    """
    请求执行器。
    负责直接与模型客户端交互，执行API请求。
    它包含了核心的请求重试、异常分类处理、模型惩罚机制和消息压缩等底层逻辑。
    """

    def __init__(
        self,
        task_name: str,
        model_set: TaskConfig,
        api_provider: APIProvider,
        client: BaseClient,
        model_info: ModelInfo,
        model_selector: ModelSelector,
    ):
        """
        初始化请求执行器。

        Args:
            task_name (str): 当前任务的名称。
            model_set (TaskConfig): 任务相关的模型配置。
            api_provider (APIProvider): API提供商配置。
            client (BaseClient): 用于发送请求的客户端实例。
            model_info (ModelInfo): 当前请求要使用的模型信息。
            model_selector (ModelSelector): 模型选择器实例，用于更新模型状态（如惩罚值）。
        """
        self.task_name = task_name
        self.model_set = model_set
        self.api_provider = api_provider
        self.client = client
        self.model_info = model_info
        self.model_selector = model_selector

    async def execute_request(
        self,
        request_type: str,
        message_list: List[Message] | None = None,
        tool_options: list[ToolOption] | None = None,
        response_format: RespFormat | None = None,
        stream_response_handler: Optional[Callable] = None,
        async_response_parser: Optional[Callable] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        embedding_input: str = "",
        audio_base64: str = "",
    ) -> APIResponse:
        """
        实际执行API请求，并包含完整的重试和异常处理逻辑。

        Args:
            request_type (str): 请求类型 ('response', 'embedding', 'audio')。
            message_list (List[Message] | None, optional): 消息列表。 Defaults to None.
            tool_options (list[ToolOption] | None, optional): 工具选项。 Defaults to None.
            response_format (RespFormat | None, optional): 响应格式要求。 Defaults to None.
            stream_response_handler (Optional[Callable], optional): 流式响应处理器。 Defaults to None.
            async_response_parser (Optional[Callable], optional): 异步响应解析器。 Defaults to None.
            temperature (Optional[float], optional): 温度参数。 Defaults to None.
            max_tokens (Optional[int], optional): 最大token数。 Defaults to None.
            embedding_input (str, optional): embedding输入文本。 Defaults to "".
            audio_base64 (str, optional): 音频base64数据。 Defaults to "".

        Returns:
            APIResponse: 从模型客户端返回的API响应对象。
            
        Raises:
            ValueError: 如果请求类型未知。
            RuntimeError: 如果所有重试都失败。
        """
        retry_remain = self.api_provider.max_retry
        compressed_messages: Optional[List[Message]] = None
        
        # 循环进行重试
        while retry_remain > 0:
            try:
                # 根据请求类型调用不同的客户端方法
                if request_type == "response":
                    assert message_list is not None, "message_list cannot be None for response requests"
                    return await self.client.get_response(
                        model_info=self.model_info,
                        message_list=(compressed_messages or message_list),
                        tool_options=tool_options,
                        max_tokens=self.model_set.max_tokens if max_tokens is None else max_tokens,
                        temperature=self.model_set.temperature if temperature is None else temperature,
                        response_format=response_format,
                        stream_response_handler=stream_response_handler,
                        async_response_parser=async_response_parser,
                        extra_params=self.model_info.extra_params,
                    )
                elif request_type == "embedding":
                    assert embedding_input, "embedding_input cannot be empty for embedding requests"
                    return await self.client.get_embedding(
                        model_info=self.model_info,
                        embedding_input=embedding_input,
                        extra_params=self.model_info.extra_params,
                    )
                elif request_type == "audio":
                    assert audio_base64 is not None, "audio_base64 cannot be None for audio requests"
                    return await self.client.get_audio_transcriptions(
                        model_info=self.model_info,
                        audio_base64=audio_base64,
                        extra_params=self.model_info.extra_params,
                    )
                raise ValueError(f"未知的请求类型: {request_type}")
            except Exception as e:
                logger.debug(f"请求失败: {str(e)}")
                # 对失败的模型应用惩罚
                self._apply_penalty_on_failure(e)

                # 使用默认异常处理器来决定下一步操作（等待、重试、压缩或终止）
                wait_interval, compressed_messages = self._default_exception_handler(
                    e,
                    remain_try=retry_remain,
                    retry_interval=self.api_provider.retry_interval,
                    messages=(message_list, compressed_messages is not None) if message_list else None,
                )

                if wait_interval == -1:
                    retry_remain = 0  # 处理器决定不再重试
                elif wait_interval > 0:
                    logger.info(f"等待 {wait_interval} 秒后重试...")
                    await asyncio.sleep(wait_interval)
            finally:
                retry_remain -= 1

        # 所有重试次数用尽后
        self.model_selector.decrease_usage_penalty(self.model_info.name) # 减少因使用而增加的基础惩罚
        logger.error(f"模型 '{self.model_info.name}' 请求失败，达到最大重试次数 {self.api_provider.max_retry} 次")
        raise RuntimeError("请求失败，已达到最大重试次数")

    def _apply_penalty_on_failure(self, e: Exception):
        """
        根据异常类型，动态调整失败模型的惩罚值。
        关键错误（如网络问题、服务器5xx错误）会施加更重的惩罚。
        """
        CRITICAL_PENALTY_MULTIPLIER = 5
        default_penalty_increment = 1
        penalty_increment = default_penalty_increment

        # 对严重错误施加更高的惩罚
        if isinstance(e, (NetworkConnectionError, ReqAbortException)):
            penalty_increment = CRITICAL_PENALTY_MULTIPLIER
        elif isinstance(e, RespNotOkException):
            if e.status_code >= 500: # 服务器内部错误
                penalty_increment = CRITICAL_PENALTY_MULTIPLIER

        # 记录日志
        log_message = f"发生未知异常: {type(e).__name__}，增加基础惩罚值: {penalty_increment}"
        if isinstance(e, (NetworkConnectionError, ReqAbortException)):
            log_message = f"发生关键错误 ({type(e).__name__})，增加惩罚值: {penalty_increment}"
        elif isinstance(e, RespNotOkException):
            log_message = f"发生响应错误 (状态码: {e.status_code})，增加惩罚值: {penalty_increment}"
        logger.warning(f"模型 '{self.model_info.name}' {log_message}")

        # 更新模型的惩罚值
        self.model_selector.update_model_penalty(self.model_info.name, penalty_increment)

    def _default_exception_handler(
        self,
        e: Exception,
        remain_try: int,
        retry_interval: int = 10,
        messages: Tuple[List[Message], bool] | None = None,
    ) -> Tuple[int, List[Message] | None]:
        """
        默认的异常分类处理器。
        根据异常类型决定是否重试、等待多久以及是否需要压缩消息。
        
        Returns:
            Tuple[int, List[Message] | None]:
                - 等待时间（秒）。-1表示不重试。
                - 压缩后的消息列表（如果有）。
        """
        model_name = self.model_info.name

        if isinstance(e, NetworkConnectionError):
            return self._check_retry(
                remain_try,
                retry_interval,
                can_retry_msg=f"任务-'{self.task_name}' 模型-'{model_name}': 连接异常，将于{retry_interval}秒后重试",
                cannot_retry_msg=f"任务-'{self.task_name}' 模型-'{model_name}': 连接异常，超过最大重试次数",
            )
        elif isinstance(e, ReqAbortException):
            logger.warning(f"任务-'{self.task_name}' 模型-'{model_name}': 请求被中断，详细信息-{str(e.message)}")
            return -1, None # 请求被中断，不重试
        elif isinstance(e, RespNotOkException):
            return self._handle_resp_not_ok(e, remain_try, retry_interval, messages)
        elif isinstance(e, RespParseException):
            logger.error(f"任务-'{self.task_name}' 模型-'{model_name}': 响应解析错误，错误信息-{e.message}")
            logger.debug(f"附加内容: {str(e.ext_info)}")
            return -1, None # 解析错误通常不可重试
        else:
            logger.error(f"任务-'{self.task_name}' 模型-'{model_name}': 未知异常，错误信息-{str(e)}")
            return -1, None # 未知异常，不重试

    def _handle_resp_not_ok(
        self,
        e: RespNotOkException,
        remain_try: int,
        retry_interval: int = 10,
        messages: tuple[list[Message], bool] | None = None,
    ) -> Tuple[int, Optional[List[Message]]]:
        """处理HTTP状态码非200的异常。"""
        model_name = self.model_info.name
        # 客户端错误 (4xx)，通常不可重试
        if e.status_code in [400, 401, 402, 403, 404]:
            logger.warning(f"任务-'{self.task_name}' 模型-'{model_name}': 请求失败，错误代码-{e.status_code}，错误信息-{e.message}")
            return -1, None
        # 请求体过大 (413)
        elif e.status_code == 413:
            # 如果消息存在且尚未被压缩，尝试压缩后重试一次
            if messages and not messages[1]: # messages[1] is a flag indicating if it's already compressed
                return self._check_retry(
                    remain_try, 0, # 立即重试
                    can_retry_msg=f"任务-'{self.task_name}' 模型-'{model_name}': 请求体过大，尝试压缩消息后重试",
                    cannot_retry_msg=f"任务-'{self.task_name}' 模型-'{model_name}': 请求体过大，压缩后仍失败",
                    can_retry_callable=compress_messages, messages=messages[0],
                )
            logger.warning(f"任务-'{self.task_name}' 模型-'{model_name}': 请求体过大，无法压缩，放弃请求。")
            return -1, None
        # 请求过于频繁 (429)
        elif e.status_code == 429:
            return self._check_retry(
                remain_try, retry_interval,
                can_retry_msg=f"任务-'{self.task_name}' 模型-'{model_name}': 请求过于频繁，将于{retry_interval}秒后重试",
                cannot_retry_msg=f"任务-'{self.task_name}' 模型-'{model_name}': 请求过于频繁，超过最大重试次数",
            )
        # 服务器错误 (5xx)，可以重试
        elif e.status_code >= 500:
            return self._check_retry(
                remain_try, retry_interval,
                can_retry_msg=f"任务-'{self.task_name}' 模型-'{model_name}': 服务器错误，将于{retry_interval}秒后重试",
                cannot_retry_msg=f"任务-'{self.task_name}' 模型-'{model_name}': 服务器错误，超过最大重试次数",
            )
        else:
            logger.warning(f"任务-'{self.task_name}' 模型-'{model_name}': 未知错误，错误代码-{e.status_code}，错误信息-{e.message}")
            return -1, None

    @staticmethod
    def _check_retry(
        remain_try: int,
        retry_interval: int,
        can_retry_msg: str,
        cannot_retry_msg: str,
        can_retry_callable: Callable | None = None,
        **kwargs,
    ) -> Tuple[int, List[Message] | None]:
        """
        辅助函数：检查是否可以重试，并执行可选的回调函数（如消息压缩）。
        """
        if remain_try > 0:
            logger.warning(f"{can_retry_msg}")
            # 如果有可执行的回调（例如压缩函数），执行它并返回结果
            if can_retry_callable is not None:
                return retry_interval, can_retry_callable(**kwargs)
            return retry_interval, None
        else:
            logger.warning(f"{cannot_retry_msg}")
            return -1, None