import re
import asyncio
import time

from enum import Enum
from rich.traceback import install
from typing import Tuple, List, Dict, Optional, Callable, Any

from src.common.logger import get_logger
from src.config.config import model_config
from src.config.api_ada_configs import APIProvider, ModelInfo, TaskConfig
from .payload_content.message import MessageBuilder, Message
from .payload_content.resp_format import RespFormat
from .payload_content.tool_option import ToolOption, ToolCall, ToolOptionBuilder, ToolParamType
from .model_client.base_client import BaseClient, APIResponse, client_registry
from .utils import compress_messages, llm_usage_recorder
from .exceptions import NetworkConnectionError, ReqAbortException, RespNotOkException, RespParseException

install(extra_lines=3)

logger = get_logger("model_utils")

# 常见Error Code Mapping
error_code_mapping = {
    400: "参数不正确",
    401: "API key 错误，认证失败，请检查 config/model_config.toml 中的配置是否正确",
    402: "账号余额不足",
    403: "需要实名,或余额不足",
    404: "Not Found",
    429: "请求过于频繁，请稍后再试",
    500: "服务器内部故障",
    503: "服务器负载过高",
}


def _normalize_image_format(image_format: str) -> str:
    """
    标准化图片格式名称，确保与各种API的兼容性
    
    Args:
        image_format (str): 原始图片格式
        
    Returns:
        str: 标准化后的图片格式
    """
    format_mapping = {
        "jpg": "jpeg",
        "JPG": "jpeg", 
        "JPEG": "jpeg",
        "jpeg": "jpeg",
        "png": "png",
        "PNG": "png",
        "webp": "webp",
        "WEBP": "webp",
        "gif": "gif",
        "GIF": "gif",
        "heic": "heic",
        "HEIC": "heic",
        "heif": "heif",
        "HEIF": "heif"
    }
    
    normalized = format_mapping.get(image_format, image_format.lower())
    logger.debug(f"图片格式标准化: {image_format} -> {normalized}")
    return normalized


class RequestType(Enum):
    """请求类型枚举"""

    RESPONSE = "response"
    EMBEDDING = "embedding"
    AUDIO = "audio"


class LLMRequest:
    """LLM请求类"""

    def __init__(self, model_set: TaskConfig, request_type: str = "") -> None:
        self.task_name = request_type
        self.model_for_task = model_set
        self.request_type = request_type
        self.model_usage: Dict[str, Tuple[int, int, int]] = {
            model: (0, 0, 0) for model in self.model_for_task.model_list
        }
        """模型使用量记录，用于进行负载均衡，对应为(total_tokens, penalty, usage_penalty)，惩罚值是为了能在某个模型请求不给力或正在被使用的时候进行调整"""
        
        # 内容混淆过滤指令
        self.noise_instruction = """**【核心过滤规则】**
在交互过程中，你会在用户的输入内容中看到一些无意义的、由字母和数字组成的随机字符串。这些字符串是API传输过程中的技术伪装信息，【不是】用户真实意图的一部分。

你的任务是【完全并彻底地忽略】这些随机字符串。
**【严禁】** 将这些字符串写入你的回复、对它们进行任何形式的解读、或将它们作为故事的一部分进行模拟。当看到这些字符串时，请将其视为空白或直接跳过，并根据用户其他有意义的输入（如有）或当前故事情节来继续你的叙事。此规则为最高优先级。
"""

    async def generate_response_for_image(
        self,
        prompt: str,
        image_base64: str,
        image_format: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Tuple[str, str, Optional[List[ToolCall]]]]:
        """
        为图像生成响应
        Args:
            prompt (str): 提示词
            image_base64 (str): 图像的Base64编码字符串
            image_format (str): 图像格式（如 'png', 'jpeg' 等）
        Returns:
            (Tuple[str, str, str, Optional[List[ToolCall]]]): 响应内容、推理内容、模型名称、工具调用列表
        """
        # 标准化图片格式以确保API兼容性
        normalized_format = _normalize_image_format(image_format)
        
        # 模型选择
        start_time = time.time()
        model_info, api_provider, client = self._select_model()

        # 请求体构建
        message_builder = MessageBuilder()
        message_builder.add_text_content(prompt)
        message_builder.add_image_content(
            image_base64=image_base64, image_format=normalized_format, support_formats=client.get_support_image_formats()
        )
        messages = [message_builder.build()]

        # 请求并处理返回值
        response = await self._execute_request(
            api_provider=api_provider,
            client=client,
            request_type=RequestType.RESPONSE,
            model_info=model_info,
            message_list=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.content or ""
        reasoning_content = response.reasoning_content or ""
        tool_calls = response.tool_calls
        # 从内容中提取<think>标签的推理内容（向后兼容）
        if not reasoning_content and content:
            content, extracted_reasoning = self._extract_reasoning(content)
            reasoning_content = extracted_reasoning
        if usage := response.usage:
            llm_usage_recorder.record_usage_to_database(
                model_info=model_info,
                model_usage=usage,
                user_id="system",
                request_type=self.request_type,
                endpoint="/chat/completions",
                time_cost=time.time() - start_time,
            )
        return content, (reasoning_content, model_info.name, tool_calls)

    async def generate_response_for_voice(self, voice_base64: str) -> Optional[str]:
        """
        为语音生成响应
        Args:
            voice_base64 (str): 语音的Base64编码字符串
        Returns:
            (Optional[str]): 生成的文本描述或None
        """
        # 模型选择
        model_info, api_provider, client = self._select_model()

        # 请求并处理返回值
        response = await self._execute_request(
            api_provider=api_provider,
            client=client,
            request_type=RequestType.AUDIO,
            model_info=model_info,
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
        异步生成响应
        Args:
            prompt (str): 提示词
            temperature (float, optional): 温度参数
            max_tokens (int, optional): 最大token数
        Returns:
            (Tuple[str, str, str, Optional[List[ToolCall]]]): 响应内容、推理内容、模型名称、工具调用列表
        """
        # 请求体构建
        start_time = time.time()
        
        # 模型选择
        model_info, api_provider, client = self._select_model()
        
        # 🔥 内容混淆处理
        processed_prompt = self._apply_content_obfuscation(prompt, api_provider)
        
        message_builder = MessageBuilder()
        message_builder.add_text_content(processed_prompt)
        messages = [message_builder.build()]
        
        tool_built = self._build_tool_options(tools)
        
        # 请求并处理返回值
        logger.debug(f"LLM选择耗时: {model_info.name} {time.time() - start_time}")
        
        # 空回复重试逻辑
        empty_retry_count = 0
        max_empty_retry = api_provider.max_retry
        empty_retry_interval = api_provider.retry_interval
        
        while empty_retry_count <= max_empty_retry:
            try:
                response = await self._execute_request(
                    api_provider=api_provider,
                    client=client,
                    request_type=RequestType.RESPONSE,
                    model_info=model_info,
                    message_list=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tool_options=tool_built,
                )
                
                content = response.content
                reasoning_content = response.reasoning_content or ""
                tool_calls = response.tool_calls
                
                # 从内容中提取<think>标签的推理内容（向后兼容）
                if not reasoning_content and content:
                    content, extracted_reasoning = self._extract_reasoning(content)
                    reasoning_content = extracted_reasoning
                
                # 检测是否为空回复
                is_empty_reply = not content or content.strip() == ""
                
                if is_empty_reply and empty_retry_count < max_empty_retry:
                    empty_retry_count += 1
                    logger.warning(f"检测到空回复，正在进行第 {empty_retry_count}/{max_empty_retry} 次重新生成")
                    
                    # 等待一定时间后重试
                    if empty_retry_interval > 0:
                        await asyncio.sleep(empty_retry_interval)
                    
                    # 重新选择模型（可能选择不同的模型）
                    model_info, api_provider, client = self._select_model()
                    continue
                
                # 记录使用情况
                if usage := response.usage:
                    llm_usage_recorder.record_usage_to_database(
                        model_info=model_info,
                        model_usage=usage,
                        user_id="system",
                        request_type=self.request_type,
                        endpoint="/chat/completions",
                    )
                
                # 如果内容仍然为空
                if not content:
                    if raise_when_empty:
                        logger.warning(f"经过 {empty_retry_count} 次重试后仍然生成空回复")
                        raise RuntimeError(f"经过 {empty_retry_count} 次重试后仍然生成空回复")
                    content = "生成的响应为空，请检查模型配置或输入内容是否正确"
                else:
                    # 成功生成非空回复
                    if empty_retry_count > 0:
                        logger.info(f"经过 {empty_retry_count} 次重试后成功生成回复")

                return content, (reasoning_content, model_info.name, tool_calls)
                
            except Exception as e:
                # 如果是网络错误等其他异常，不进行空回复重试
                if empty_retry_count == 0:  # 只在第一次出错时抛出异常
                    raise e
                else:
                    # 如果已经在重试过程中出错，记录日志并继续
                    logger.error(f"重试过程中出错: {e}")
                    empty_retry_count += 1
                    if empty_retry_count <= max_empty_retry and empty_retry_interval > 0:
                        await asyncio.sleep(empty_retry_interval)
                    continue
        
        # 如果所有重试都失败了
        if raise_when_empty:
            raise RuntimeError(f"经过 {max_empty_retry} 次重试后仍然无法生成有效回复")
        return "生成的响应为空，请检查模型配置或输入内容是否正确", ("", model_info.name, None)

    async def get_embedding(self, embedding_input: str) -> Tuple[List[float], str]:
        """获取嵌入向量
        Args:
            embedding_input (str): 获取嵌入的目标
        Returns:
            (Tuple[List[float], str]): (嵌入向量，使用的模型名称)
        """
        # 无需构建消息体，直接使用输入文本
        start_time = time.time()
        model_info, api_provider, client = self._select_model()

        # 请求并处理返回值
        response = await self._execute_request(
            api_provider=api_provider,
            client=client,
            request_type=RequestType.EMBEDDING,
            model_info=model_info,
            embedding_input=embedding_input,
        )

        embedding = response.embedding

        if usage := response.usage:
            llm_usage_recorder.record_usage_to_database(
                model_info=model_info,
                model_usage=usage,
                user_id="system",
                request_type=self.request_type,
                endpoint="/embeddings",
                time_cost=time.time() - start_time,
            )

        if not embedding:
            raise RuntimeError("获取embedding失败")

        return embedding, model_info.name

    def _select_model(self) -> Tuple[ModelInfo, APIProvider, BaseClient]:
        """
        根据总tokens和惩罚值选择的模型
        """
        least_used_model_name = min(
            self.model_usage,
            key=lambda k: self.model_usage[k][0] + self.model_usage[k][1] * 300 + self.model_usage[k][2] * 1000,
        )
        model_info = model_config.get_model_info(least_used_model_name)
        api_provider = model_config.get_provider(model_info.api_provider)
        client = client_registry.get_client_class_instance(api_provider)
        logger.debug(f"选择请求模型: {model_info.name}")
        total_tokens, penalty, usage_penalty = self.model_usage[model_info.name]
        self.model_usage[model_info.name] = (total_tokens, penalty, usage_penalty + 1)  # 增加使用惩罚值防止连续使用
        return model_info, api_provider, client

    async def _execute_request(
        self,
        api_provider: APIProvider,
        client: BaseClient,
        request_type: RequestType,
        model_info: ModelInfo,
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
        实际执行请求的方法

        包含了重试和异常处理逻辑
        """
        retry_remain = api_provider.max_retry
        compressed_messages: Optional[List[Message]] = None
        while retry_remain > 0:
            try:
                if request_type == RequestType.RESPONSE:
                    assert message_list is not None, "message_list cannot be None for response requests"
                    return await client.get_response(
                        model_info=model_info,
                        message_list=(compressed_messages or message_list),
                        tool_options=tool_options,
                        max_tokens=self.model_for_task.max_tokens if max_tokens is None else max_tokens,
                        temperature=self.model_for_task.temperature if temperature is None else temperature,
                        response_format=response_format,
                        stream_response_handler=stream_response_handler,
                        async_response_parser=async_response_parser,
                        extra_params=model_info.extra_params,
                    )
                elif request_type == RequestType.EMBEDDING:
                    assert embedding_input, "embedding_input cannot be empty for embedding requests"
                    return await client.get_embedding(
                        model_info=model_info,
                        embedding_input=embedding_input,
                        extra_params=model_info.extra_params,
                    )
                elif request_type == RequestType.AUDIO:
                    assert audio_base64 is not None, "audio_base64 cannot be None for audio requests"
                    return await client.get_audio_transcriptions(
                        model_info=model_info,
                        audio_base64=audio_base64,
                        extra_params=model_info.extra_params,
                    )
            except Exception as e:
                logger.debug(f"请求失败: {str(e)}")
                # 处理异常
                total_tokens, penalty, usage_penalty = self.model_usage[model_info.name]
                self.model_usage[model_info.name] = (total_tokens, penalty + 1, usage_penalty)

                wait_interval, compressed_messages = self._default_exception_handler(
                    e,
                    self.task_name,
                    model_name=model_info.name,
                    remain_try=retry_remain,
                    retry_interval=api_provider.retry_interval,
                    messages=(message_list, compressed_messages is not None) if message_list else None,
                )

                if wait_interval == -1:
                    retry_remain = 0  # 不再重试
                elif wait_interval > 0:
                    logger.info(f"等待 {wait_interval} 秒后重试...")
                    await asyncio.sleep(wait_interval)
            finally:
                # 放在finally防止死循环
                retry_remain -= 1
        total_tokens, penalty, usage_penalty = self.model_usage[model_info.name]
        self.model_usage[model_info.name] = (total_tokens, penalty, usage_penalty - 1)  # 使用结束，减少使用惩罚值
        logger.error(f"模型 '{model_info.name}' 请求失败，达到最大重试次数 {api_provider.max_retry} 次")
        raise RuntimeError("请求失败，已达到最大重试次数")

    def _default_exception_handler(
        self,
        e: Exception,
        task_name: str,
        model_name: str,
        remain_try: int,
        retry_interval: int = 10,
        messages: Tuple[List[Message], bool] | None = None,
    ) -> Tuple[int, List[Message] | None]:
        """
        默认异常处理函数
        Args:
            e (Exception): 异常对象
            task_name (str): 任务名称
            model_name (str): 模型名称
            remain_try (int): 剩余尝试次数
            retry_interval (int): 重试间隔
            messages (tuple[list[Message], bool] | None): (消息列表, 是否已压缩过)
        Returns:
            (等待间隔（如果为0则不等待，为-1则不再请求该模型）, 新的消息列表（适用于压缩消息）)
        """

        if isinstance(e, NetworkConnectionError):  # 网络连接错误
            return self._check_retry(
                remain_try,
                retry_interval,
                can_retry_msg=f"任务-'{task_name}' 模型-'{model_name}': 连接异常，将于{retry_interval}秒后重试",
                cannot_retry_msg=f"任务-'{task_name}' 模型-'{model_name}': 连接异常，超过最大重试次数，请检查网络连接状态或URL是否正确",
            )
        elif isinstance(e, ReqAbortException):
            logger.warning(f"任务-'{task_name}' 模型-'{model_name}': 请求被中断，详细信息-{str(e.message)}")
            return -1, None  # 不再重试请求该模型
        elif isinstance(e, RespNotOkException):
            return self._handle_resp_not_ok(
                e,
                task_name,
                model_name,
                remain_try,
                retry_interval,
                messages,
            )
        elif isinstance(e, RespParseException):
            # 响应解析错误
            logger.error(f"任务-'{task_name}' 模型-'{model_name}': 响应解析错误，错误信息-{e.message}")
            logger.debug(f"附加内容: {str(e.ext_info)}")
            return -1, None  # 不再重试请求该模型
        else:
            logger.error(f"任务-'{task_name}' 模型-'{model_name}': 未知异常，错误信息-{str(e)}")
            return -1, None  # 不再重试请求该模型

    def _check_retry(
        self,
        remain_try: int,
        retry_interval: int,
        can_retry_msg: str,
        cannot_retry_msg: str,
        can_retry_callable: Callable | None = None,
        **kwargs,
    ) -> Tuple[int, List[Message] | None]:
        """辅助函数：检查是否可以重试
        Args:
            remain_try (int): 剩余尝试次数
            retry_interval (int): 重试间隔
            can_retry_msg (str): 可以重试时的提示信息
            cannot_retry_msg (str): 不可以重试时的提示信息
            can_retry_callable (Callable | None): 可以重试时调用的函数（如果有）
            **kwargs: 其他参数

        Returns:
            (Tuple[int, List[Message] | None]): (等待间隔（如果为0则不等待，为-1则不再请求该模型）, 新的消息列表（适用于压缩消息）)
        """
        if remain_try > 0:
            # 还有重试机会
            logger.warning(f"{can_retry_msg}")
            if can_retry_callable is not None:
                return retry_interval, can_retry_callable(**kwargs)
            else:
                return retry_interval, None
        else:
            # 达到最大重试次数
            logger.warning(f"{cannot_retry_msg}")
            return -1, None  # 不再重试请求该模型

    def _handle_resp_not_ok(
        self,
        e: RespNotOkException,
        task_name: str,
        model_name: str,
        remain_try: int,
        retry_interval: int = 10,
        messages: tuple[list[Message], bool] | None = None,
    ):
        """
        处理响应错误异常
        Args:
            e (RespNotOkException): 响应错误异常对象
            task_name (str): 任务名称
            model_name (str): 模型名称
            remain_try (int): 剩余尝试次数
            retry_interval (int): 重试间隔
            messages (tuple[list[Message], bool] | None): (消息列表, 是否已压缩过)
        Returns:
            (等待间隔（如果为0则不等待，为-1则不再请求该模型）, 新的消息列表（适用于压缩消息）)
        """
        # 响应错误
        if e.status_code in [400, 401, 402, 403, 404]:
            # 客户端错误
            logger.warning(
                f"任务-'{task_name}' 模型-'{model_name}': 请求失败，错误代码-{e.status_code}，错误信息-{e.message}"
            )
            return -1, None  # 不再重试请求该模型
        elif e.status_code == 413:
            if messages and not messages[1]:
                # 消息列表不为空且未压缩，尝试压缩消息
                return self._check_retry(
                    remain_try,
                    0,
                    can_retry_msg=f"任务-'{task_name}' 模型-'{model_name}': 请求体过大，尝试压缩消息后重试",
                    cannot_retry_msg=f"任务-'{task_name}' 模型-'{model_name}': 请求体过大，压缩消息后仍然过大，放弃请求",
                    can_retry_callable=compress_messages,
                    messages=messages[0],
                )
            # 没有消息可压缩
            logger.warning(f"任务-'{task_name}' 模型-'{model_name}': 请求体过大，无法压缩消息，放弃请求。")
            return -1, None
        elif e.status_code == 429:
            # 请求过于频繁
            return self._check_retry(
                remain_try,
                retry_interval,
                can_retry_msg=f"任务-'{task_name}' 模型-'{model_name}': 请求过于频繁，将于{retry_interval}秒后重试",
                cannot_retry_msg=f"任务-'{task_name}' 模型-'{model_name}': 请求过于频繁，超过最大重试次数，放弃请求",
            )
        elif e.status_code >= 500:
            # 服务器错误
            return self._check_retry(
                remain_try,
                retry_interval,
                can_retry_msg=f"任务-'{task_name}' 模型-'{model_name}': 服务器错误，将于{retry_interval}秒后重试",
                cannot_retry_msg=f"任务-'{task_name}' 模型-'{model_name}': 服务器错误，超过最大重试次数，请稍后再试",
            )
        else:
            # 未知错误
            logger.warning(
                f"任务-'{task_name}' 模型-'{model_name}': 未知错误，错误代码-{e.status_code}，错误信息-{e.message}"
            )
            return -1, None

    def _build_tool_options(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[ToolOption]]:
        # sourcery skip: extract-method
        """构建工具选项列表"""
        if not tools:
            return None
        tool_options: List[ToolOption] = []
        for tool in tools:
            tool_legal = True
            tool_options_builder = ToolOptionBuilder()
            tool_options_builder.set_name(tool.get("name", ""))
            tool_options_builder.set_description(tool.get("description", ""))
            parameters: List[Tuple[str, str, str, bool, List[str] | None]] = tool.get("parameters", [])
            for param in parameters:
                try:
                    assert isinstance(param, tuple) and len(param) == 5, "参数必须是包含5个元素的元组"
                    assert isinstance(param[0], str), "参数名称必须是字符串"
                    assert isinstance(param[1], ToolParamType), "参数类型必须是ToolParamType枚举"
                    assert isinstance(param[2], str), "参数描述必须是字符串"
                    assert isinstance(param[3], bool), "参数是否必填必须是布尔值"
                    assert isinstance(param[4], list) or param[4] is None, "参数枚举值必须是列表或None"
                    tool_options_builder.add_param(
                        name=param[0],
                        param_type=param[1],
                        description=param[2],
                        required=param[3],
                        enum_values=param[4],
                    )
                except AssertionError as ae:
                    tool_legal = False
                    logger.error(f"{param[0]} 参数定义错误: {str(ae)}")
                except Exception as e:
                    tool_legal = False
                    logger.error(f"构建工具参数失败: {str(e)}")
            if tool_legal:
                tool_options.append(tool_options_builder.build())
        return tool_options or None

    @staticmethod
    def _extract_reasoning(content: str) -> Tuple[str, str]:
        """CoT思维链提取，向后兼容"""
        match = re.search(r"(?:<think>)?(.*?)</think>", content, re.DOTALL)
        content = re.sub(r"(?:<think>)?.*?</think>", "", content, flags=re.DOTALL, count=1).strip()
        reasoning = match[1].strip() if match else ""
        return content, reasoning

    def _apply_content_obfuscation(self, text: str, api_provider) -> str:
        """根据API提供商配置对文本进行混淆处理"""
        if not hasattr(api_provider, 'enable_content_obfuscation') or not api_provider.enable_content_obfuscation:
            logger.debug(f"API提供商 '{api_provider.name}' 未启用内容混淆")
            return text
        
        intensity = getattr(api_provider, 'obfuscation_intensity', 1)
        logger.info(f"为API提供商 '{api_provider.name}' 启用内容混淆，强度级别: {intensity}")
        
        # 在开头加入过滤规则指令
        processed_text = self.noise_instruction + "\n\n" + text
        logger.debug(f"已添加过滤规则指令，文本长度: {len(text)} -> {len(processed_text)}")
        
        # 添加随机乱码
        final_text = self._inject_random_noise(processed_text, intensity)
        logger.debug(f"乱码注入完成，最终文本长度: {len(final_text)}")
        
        return final_text
    
    def _inject_random_noise(self, text: str, intensity: int) -> str:
        """在文本中注入随机乱码"""
        import random
        import string
        
        def generate_noise(length: int) -> str:
            """生成指定长度的随机乱码字符"""
            chars = (
                string.ascii_letters +           # a-z, A-Z
                string.digits +                  # 0-9
                '!@#$%^&*()_+-=[]{}|;:,.<>?' +  # 特殊符号
                '一二三四五六七八九零壹贰叁' +      # 中文字符
                'αβγδεζηθικλμνξοπρστυφχψω' +     # 希腊字母
                '∀∃∈∉∪∩⊂⊃∧∨¬→↔∴∵'            # 数学符号
            )
            return ''.join(random.choice(chars) for _ in range(length))
        
        # 强度参数映射
        params = {
            1: {"probability": 15, "length": (3, 6)},     # 低强度：15%概率，3-6个字符
            2: {"probability": 25, "length": (5, 10)},    # 中强度：25%概率，5-10个字符
            3: {"probability": 35, "length": (8, 15)}     # 高强度：35%概率，8-15个字符
        }
        
        config = params.get(intensity, params[1])
        logger.debug(f"乱码注入参数: 概率={config['probability']}%, 长度范围={config['length']}")
        
        # 按词分割处理
        words = text.split()
        result = []
        noise_count = 0
        
        for word in words:
            result.append(word)
            # 根据概率插入乱码
            if random.randint(1, 100) <= config["probability"]:
                noise_length = random.randint(*config["length"])
                noise = generate_noise(noise_length)
                result.append(noise)
                noise_count += 1
        
        logger.debug(f"共注入 {noise_count} 个乱码片段，原词数: {len(words)}")
        return ' '.join(result)
