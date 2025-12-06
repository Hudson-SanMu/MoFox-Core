#!/usr/bin/env python3
"""
AWS Bedrock 客户端测试脚本
测试 BedrockClient 的基本功能
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config.api_ada_configs import APIProvider, ModelInfo
from src.llm_models.model_client.bedrock_client import BedrockClient
from src.llm_models.payload_content.message import MessageBuilder


async def test_basic_conversation():
    """测试基本对话功能"""
    print("=" * 60)
    print("测试 1: 基本对话功能")
    print("=" * 60)

    # 配置 API Provider（请替换为你的真实凭证）
    provider = APIProvider(
        name="bedrock_test",
        base_url="",  # Bedrock 不需要
        api_key="YOUR_AWS_ACCESS_KEY_ID",  # 替换为你的 AWS Access Key
        client_type="bedrock",
        max_retry=2,
        timeout=60,
        retry_interval=10,
        extra_params={
            "aws_secret_key": "YOUR_AWS_SECRET_ACCESS_KEY",  # 替换为你的 AWS Secret Key
            "region": "us-east-1",
        },
    )

    # 配置模型信息
    model = ModelInfo(
        model_identifier="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        name="claude-3.5-sonnet-bedrock",
        api_provider="bedrock_test",
        price_in=3.0,
        price_out=15.0,
        force_stream_mode=False,
    )

    # 创建客户端
    client = BedrockClient(provider)

    # 构建消息
    builder = MessageBuilder()
    builder.add_user_message("你好！请用一句话介绍 AWS Bedrock。")

    try:
        # 发送请求
        response = await client.get_response(
            model_info=model, message_list=[builder.build()], max_tokens=200, temperature=0.7
        )

        print(f"✅ 响应内容: {response.content}")
        if response.usage:
            print(
                f"📊 Token 使用: 输入={response.usage.prompt_tokens}, "
                f"输出={response.usage.completion_tokens}, "
                f"总计={response.usage.total_tokens}"
            )
        print("\n测试通过！✅\n")
    except Exception as e:
        print(f"❌ 测试失败: {e!s}")
        import traceback

        traceback.print_exc()


async def test_streaming():
    """测试流式输出功能"""
    print("=" * 60)
    print("测试 2: 流式输出功能")
    print("=" * 60)

    provider = APIProvider(
        name="bedrock_test",
        base_url="",
        api_key="YOUR_AWS_ACCESS_KEY_ID",
        client_type="bedrock",
        max_retry=2,
        timeout=60,
        extra_params={
            "aws_secret_key": "YOUR_AWS_SECRET_ACCESS_KEY",
            "region": "us-east-1",
        },
    )

    model = ModelInfo(
        model_identifier="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        name="claude-3.5-sonnet-bedrock",
        api_provider="bedrock_test",
        price_in=3.0,
        price_out=15.0,
        force_stream_mode=True,  # 启用流式模式
    )

    client = BedrockClient(provider)
    builder = MessageBuilder()
    builder.add_user_message("写一个关于人工智能的三行诗。")

    try:
        print("🔄 流式响应中...")
        response = await client.get_response(
            model_info=model, message_list=[builder.build()], max_tokens=100, temperature=0.7
        )

        print(f"✅ 完整响应: {response.content}")
        print("\n测试通过！✅\n")
    except Exception as e:
        print(f"❌ 测试失败: {e!s}")


async def test_multimodal():
    """测试多模态（图片输入）功能"""
    print("=" * 60)
    print("测试 3: 多模态功能（需要准备图片）")
    print("=" * 60)
    print("⏭️  跳过（需要实际图片文件）\n")


async def test_tool_calling():
    """测试工具调用功能"""
    print("=" * 60)
    print("测试 4: 工具调用功能")
    print("=" * 60)

    from src.llm_models.payload_content.tool_option import ToolOption, ToolOptionBuilder, ToolParamType

    provider = APIProvider(
        name="bedrock_test",
        base_url="",
        api_key="YOUR_AWS_ACCESS_KEY_ID",
        client_type="bedrock",
        extra_params={
            "aws_secret_key": "YOUR_AWS_SECRET_ACCESS_KEY",
            "region": "us-east-1",
        },
    )

    model = ModelInfo(
        model_identifier="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        name="claude-3.5-sonnet-bedrock",
        api_provider="bedrock_test",
    )

    # 定义工具
    tool_builder = ToolOptionBuilder()
    tool_builder.set_name("get_weather").set_description("获取指定城市的天气信息").add_param(
        name="city", param_type=ToolParamType.STRING, description="城市名称", required=True
    )

    tool = tool_builder.build()

    client = BedrockClient(provider)
    builder = MessageBuilder()
    builder.add_user_message("北京今天天气怎么样？")

    try:
        response = await client.get_response(
            model_info=model, message_list=[builder.build()], tool_options=[tool], max_tokens=200
        )

        if response.tool_calls:
            print(f"✅ 模型调用了工具:")
            for call in response.tool_calls:
                print(f"  - 工具名: {call.func_name}")
                print(f"  - 参数: {call.args}")
        else:
            print(f"⚠️  模型没有调用工具，而是直接回复: {response.content}")

        print("\n测试通过！✅\n")
    except Exception as e:
        print(f"❌ 测试失败: {e!s}")


async def main():
    """主测试函数"""
    print("\n🚀 AWS Bedrock 客户端测试开始\n")
    print("⚠️  请确保已配置 AWS 凭证！")
    print("⚠️  修改脚本中的 'YOUR_AWS_ACCESS_KEY_ID' 和 'YOUR_AWS_SECRET_ACCESS_KEY'\n")

    # 运行测试
    await test_basic_conversation()
    # await test_streaming()
    # await test_multimodal()
    # await test_tool_calling()

    print("=" * 60)
    print("🎉 所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
