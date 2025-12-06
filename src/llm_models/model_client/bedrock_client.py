import asyncio
import base64
import io
import json
from collections.abc import Callable, Coroutine
from typing import Any

import aioboto3
import orjson
from botocore.config import Config
from json_repair import repair_json

from src.common.logger import get_logger
from src.config.api_ada_configs import APIProvider, ModelInfo

from ..exceptions import (
    NetworkConnectionError,
    ReqAbortException,
    RespNotOkException,
    RespParseException,
)
from ..payload_content.message import Message, RoleType
from ..payload_content.resp_format import RespFormat
from ..payload_content.tool_option import ToolCall, ToolOption, ToolParam
from .base_client import APIResponse, BaseClient, UsageRecord, client_registry

logger = get_logger("Bedrock客户端")


def _convert_messages_to_converse(messages: list[Message]) -> list[dict[str, Any]]:
    """
    转换消息格式 - 将消息转换为 Bedrock Converse API 所需的格式
    :param messages: 消息列表
    :return: 转换后的消息列表
    """

    def _convert_message_item(message: Message) -> dict[str, Any]:
        """
        转换单个消息格式
        :param message: 消息对象
        :return: 转换后的消息字典
        """
        # Bedrock Converse API 格式
        content: list[dict[str, Any]] = []

        if isinstance(message.content, str):
            content.append({"text": message.content})
        elif isinstance(message.content, list):
            for item in message.content:
                if isinstance(item, tuple):
                    # 图片格式：(format, base64_data)
                    image_format = item[0].lower()
                    image_bytes = base64.b64decode(item[1])
                    content.append(
                        {
                            "image": {
                                "format": image_format if image_format in ["png", "jpeg", "gif", "webp"] else "jpeg",
                                "source": {"bytes": image_bytes},
                            }
                        }
                    )
                elif isinstance(item, str):
                    content.append({"text": item})
        else:
            raise RuntimeError("无法触及的代码：请使用MessageBuilder类构建消息对象")

        ret = {
            "role": "user" if message.role == RoleType.User else "assistant",
            "content": content,
        }

        return ret

    # Bedrock 不支持 system 和 tool 角色，需要过滤
    converted = []
    for msg in messages:
        if msg.role in [RoleType.User, RoleType.Assistant]:
            converted.append(_convert_message_item(msg))

    return converted


def _convert_tool_options_to_bedrock(tool_options: list[ToolOption]) -> list[dict[str, Any]]:
    """
    转换工具选项格式 - 将工具选项转换为 Bedrock Converse API 所需的格式
    :param tool_options: 工具选项列表
    :return: 转换后的工具选项列表
    """

    def _convert_tool_param(tool_param: ToolParam) -> dict[str, Any]:
        """转换单个工具参数"""
        param_dict: dict[str, Any] = {
            "type": tool_param.param_type.value,
            "description": tool_param.description,
        }
        if tool_param.enum_values:
            param_dict["enum"] = tool_param.enum_values
        return param_dict

    def _convert_tool_option_item(tool_option: ToolOption) -> dict[str, Any]:
        """转换单个工具项"""
        tool_spec: dict[str, Any] = {
            "name": tool_option.name,
            "description": tool_option.description,
        }
        if tool_option.params:
            tool_spec["inputSchema"] = {
                "json": {
                    "type": "object",
                    "properties": {param.name: _convert_tool_param(param) for param in tool_option.params},
                    "required": [param.name for param in tool_option.params if param.required],
                }
            }
        return {"toolSpec": tool_spec}

    return [_convert_tool_option_item(opt) for opt in tool_options]


async def _default_stream_response_handler(
    resp_stream: Any,
    interrupt_flag: asyncio.Event | None,
) -> tuple[APIResponse, tuple[int, int, int] | None]:
    """
    流式响应处理函数 - 处理 Bedrock Converse Stream API 的响应
    :param resp_stream: 流式响应对象
    :param interrupt_flag: 中断标志
    :return: (APIResponse对象, usage元组)
    """
    _fc_delta_buffer = io.StringIO()  # 正式内容缓冲区
    _tool_calls_buffer: list[tuple[str, str, io.StringIO]] = []  # 工具调用缓冲区
    _usage_record = None

    def _insure_buffer_closed():
        if _fc_delta_buffer and not _fc_delta_buffer.closed:
            _fc_delta_buffer.close()
        for _, _, buffer in _tool_calls_buffer:
            if buffer and not buffer.closed:
                buffer.close()

    try:
        async for event in resp_stream["stream"]:
            if interrupt_flag and interrupt_flag.is_set():
                _insure_buffer_closed()
                raise ReqAbortException("请求被外部信号中断")

            # 处理内容块
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    _fc_delta_buffer.write(delta["text"])
                elif "toolUse" in delta:
                    # 工具调用
                    tool_use = delta["toolUse"]
                    if "input" in tool_use:
                        # 追加工具调用参数
                        if tool_use.get("toolUseId"):
                            # 新的工具调用
                            _tool_calls_buffer.append(
                                (
                                    tool_use["toolUseId"],
                                    tool_use.get("name", ""),
                                    io.StringIO(json.dumps(tool_use["input"])),
                                )
                            )

            # 处理元数据（包含 usage）
            if "metadata" in event:
                metadata = event["metadata"]
                if "usage" in metadata:
                    usage = metadata["usage"]
                    _usage_record = (
                        usage.get("inputTokens", 0),
                        usage.get("outputTokens", 0),
                        usage.get("totalTokens", 0),
                    )

        # 构建响应
        resp = APIResponse()
        if _fc_delta_buffer.tell() > 0:
            resp.content = _fc_delta_buffer.getvalue()
        _fc_delta_buffer.close()

        if _tool_calls_buffer:
            resp.tool_calls = []
            for call_id, function_name, arguments_buffer in _tool_calls_buffer:
                if arguments_buffer.tell() > 0:
                    raw_arg_data = arguments_buffer.getvalue()
                    arguments_buffer.close()
                    try:
                        arguments = orjson.loads(repair_json(raw_arg_data))
                        if not isinstance(arguments, dict):
                            raise RespParseException(
                                None,
                                f"响应解析失败，工具调用参数无法解析为字典类型。原始响应：\n{raw_arg_data}",
                            )
                    except orjson.JSONDecodeError as e:
                        raise RespParseException(
                            None,
                            f"响应解析失败，无法解析工具调用参数。原始响应：{raw_arg_data}",
                        ) from e
                else:
                    arguments_buffer.close()
                    arguments = None

                resp.tool_calls.append(ToolCall(call_id, function_name, args=arguments))

        return resp, _usage_record

    except Exception as e:
        _insure_buffer_closed()
        raise


async def _default_async_response_parser(
    resp_data: dict[str, Any],
) -> tuple[APIResponse, tuple[int, int, int] | None]:
    """
    默认异步响应解析函数 - 解析 Bedrock Converse API 的响应
    :param resp_data: 响应数据
    :return: (APIResponse对象, usage元组)
    """
    resp = APIResponse()

    # 解析输出内容
    if "output" in resp_data and "message" in resp_data["output"]:
        message = resp_data["output"]["message"]
        content_blocks = message.get("content", [])

        text_parts = []
        tool_calls = []

        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                tool_calls.append(
                    ToolCall(
                        call_id=tool_use.get("toolUseId", ""),
                        func_name=tool_use.get("name", ""),
                        args=tool_use.get("input", {}),
                    )
                )

        if text_parts:
            resp.content = "".join(text_parts)
        if tool_calls:
            resp.tool_calls = tool_calls

    # 解析 usage
    usage_record = None
    if "usage" in resp_data:
        usage = resp_data["usage"]
        usage_record = (
            usage.get("inputTokens", 0),
            usage.get("outputTokens", 0),
            usage.get("totalTokens", 0),
        )

    resp.raw_data = resp_data
    return resp, usage_record


@client_registry.register_client_class("bedrock")
class BedrockClient(BaseClient):
    """AWS Bedrock 客户端"""

    def __init__(self, api_provider: APIProvider):
        super().__init__(api_provider)

        # 从 extra_params 获取 AWS 配置
        # 支持两种认证方式：
        # 方式1（显式凭证）：api_key + extra_params.aws_secret_key
        # 方式2（IAM角色）：只配置 region，自动从环境/实例角色获取凭证
        region = api_provider.extra_params.get("region", "us-east-1")
        aws_secret_key = api_provider.extra_params.get("aws_secret_key")

        # 配置 boto3
        self.region = region
        self.boto_config = Config(
            region_name=self.region,
            connect_timeout=api_provider.timeout,
            read_timeout=api_provider.timeout,
            retries={"max_attempts": api_provider.max_retry, "mode": "adaptive"},
        )

        # 判断认证方式
        if aws_secret_key:
            # 方式1：显式 IAM 凭证
            self.aws_access_key_id = api_provider.get_api_key()
            self.aws_secret_access_key = aws_secret_key
            self.session = aioboto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region,
            )
            logger.info(f"初始化 Bedrock 客户端（IAM 凭证模式），区域: {self.region}")
        else:
            # 方式2：IAM 角色自动认证（从环境变量、EC2/ECS 实例角色获取）
            self.session = aioboto3.Session(region_name=self.region)
            logger.info(f"初始化 Bedrock 客户端（IAM 角色模式），区域: {self.region}")
            logger.info("将使用环境变量或实例角色自动获取 AWS 凭证")

    async def get_response(
        self,
        model_info: ModelInfo,
        message_list: list[Message],
        tool_options: list[ToolOption] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_format: RespFormat | None = None,
        stream_response_handler: Callable[[Any, asyncio.Event | None], tuple[APIResponse, tuple[int, int, int]]]
        | None = None,
        async_response_parser: Callable[[Any], tuple[APIResponse, tuple[int, int, int]]] | None = None,
        interrupt_flag: asyncio.Event | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """
        获取对话响应
        """
        try:
            # 提取 system prompt
            system_prompts = []
            filtered_messages = []
            for msg in message_list:
                if msg.role == RoleType.System:
                    if isinstance(msg.content, str):
                        system_prompts.append({"text": msg.content})
                else:
                    filtered_messages.append(msg)

            # 转换消息格式
            messages = _convert_messages_to_converse(filtered_messages)

            # 构建请求参数
            request_params: dict[str, Any] = {
                "modelId": model_info.model_identifier,
                "messages": messages,
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                },
            }

            # 添加 system prompt
            if system_prompts:
                request_params["system"] = system_prompts

            # 添加工具配置
            if tool_options:
                request_params["toolConfig"] = {"tools": _convert_tool_options_to_bedrock(tool_options)}

            # 合并额外参数
            if extra_params:
                request_params.update(extra_params)

            # 合并模型配置的额外参数
            if model_info.extra_params:
                request_params.update(model_info.extra_params)

            # 创建 Bedrock Runtime 客户端
            async with self.session.client("bedrock-runtime", config=self.boto_config) as bedrock_client:
                # 判断是否使用流式模式
                use_stream = model_info.force_stream_mode or stream_response_handler is not None

                if use_stream:
                    # 流式调用
                    response = await bedrock_client.converse_stream(**request_params)
                    if stream_response_handler:
                        # 用户提供的处理器（可能是同步的）
                        result = stream_response_handler(response, interrupt_flag)
                        if asyncio.iscoroutine(result):
                            api_resp, usage_tuple = await result
                        else:
                            api_resp, usage_tuple = result  # type: ignore
                    else:
                        # 默认异步处理器
                        api_resp, usage_tuple = await _default_stream_response_handler(response, interrupt_flag)
                else:
                    # 非流式调用
                    response = await bedrock_client.converse(**request_params)
                    if async_response_parser:
                        # 用户提供的解析器（可能是同步的）
                        result = async_response_parser(response)
                        if asyncio.iscoroutine(result):
                            api_resp, usage_tuple = await result
                        else:
                            api_resp, usage_tuple = result  # type: ignore
                    else:
                        # 默认异步解析器
                        api_resp, usage_tuple = await _default_async_response_parser(response)

                # 设置 usage
                if usage_tuple:
                    api_resp.usage = UsageRecord(
                        model_name=model_info.model_identifier,
                        provider_name=self.api_provider.name,
                        prompt_tokens=usage_tuple[0],
                        completion_tokens=usage_tuple[1],
                        total_tokens=usage_tuple[2],
                    )

                return api_resp

        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"Bedrock API 调用失败 ({error_type}): {e!s}")

            # 处理特定错误类型
            if "ThrottlingException" in error_type or "ServiceQuota" in error_type:
                raise RespNotOkException(429, f"请求限流: {e!s}") from e
            elif "ValidationException" in error_type:
                raise RespParseException(400, f"请求参数错误: {e!s}") from e
            elif "AccessDeniedException" in error_type:
                raise RespNotOkException(403, f"访问被拒绝: {e!s}") from e
            elif "ResourceNotFoundException" in error_type:
                raise RespNotOkException(404, f"模型不存在: {e!s}") from e
            elif "timeout" in str(e).lower() or "timed out" in str(e).lower():
                logger.error(f"请求超时: {e!s}")
                raise NetworkConnectionError() from e
            else:
                logger.error(f"网络连接错误: {e!s}")
                raise NetworkConnectionError() from e

    async def get_embedding(
        self,
        model_info: ModelInfo,
        embedding_input: str | list[str],
        extra_params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """
        获取文本嵌入（Bedrock 支持 Titan Embeddings 等模型）
        """
        try:
            async with self.session.client("bedrock-runtime", config=self.boto_config) as bedrock_client:
                # Bedrock Embeddings 使用 InvokeModel API
                is_batch = isinstance(embedding_input, list)
                input_text = embedding_input if is_batch else [embedding_input]

                results = []
                total_tokens = 0

                for text in input_text:
                    # 构建请求体（Titan Embeddings 格式）
                    body = json.dumps({"inputText": text})

                    response = await bedrock_client.invoke_model(
                        modelId=model_info.model_identifier,
                        contentType="application/json",
                        accept="application/json",
                        body=body,
                    )

                    # 解析响应
                    response_body = json.loads(await response["body"].read())
                    embedding = response_body.get("embedding", [])
                    results.append(embedding)

                    # 累计 token 使用
                    if "inputTokenCount" in response_body:
                        total_tokens += response_body["inputTokenCount"]

                api_resp = APIResponse()
                api_resp.embedding = results if is_batch else results[0]
                api_resp.usage = UsageRecord(
                    model_name=model_info.model_identifier,
                    provider_name=self.api_provider.name,
                    prompt_tokens=total_tokens,
                    completion_tokens=0,
                    total_tokens=total_tokens,
                )

                return api_resp

        except Exception as e:
            logger.error(f"Bedrock Embedding 调用失败: {e!s}")
            raise NetworkConnectionError() from e

    async def get_audio_transcriptions(
        self,
        model_info: ModelInfo,
        audio_base64: str,
        extra_params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """
        获取音频转录（Bedrock 暂不直接支持，抛出未实现异常）
        """
        raise NotImplementedError("AWS Bedrock 暂不支持音频转录功能，建议使用 AWS Transcribe 服务")

    def get_support_image_formats(self) -> list[str]:
        """
        获取支持的图片格式
        :return: 支持的图片格式列表
        """
        return ["png", "jpeg", "jpg", "gif", "webp"]
