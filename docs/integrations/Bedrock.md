# AWS Bedrock 集成指南

## 概述

MoFox-Bot 已完全集成 AWS Bedrock，支持使用 **Converse API** 统一调用所有 Bedrock 模型，包括：
- Amazon Nova 系列
- Anthropic Claude 3/3.5
- Meta Llama 2/3
- Mistral AI
- Cohere Command
- AI21 Jamba
- Stability AI SDXL

## 配置示例

### 1. 配置 API Provider

在 `config/model_config.toml` 中添加 Bedrock Provider：

```toml
[[api_providers]]
name = "bedrock_us_east"
base_url = ""  # Bedrock 不需要 base_url，留空即可
api_key = "YOUR_AWS_ACCESS_KEY_ID"  # AWS Access Key ID
client_type = "bedrock"
max_retry = 2
timeout = 60
retry_interval = 10

[api_providers.extra_params]
aws_secret_key = "YOUR_AWS_SECRET_ACCESS_KEY"  # AWS Secret Access Key
region = "us-east-1"  # AWS 区域，默认 us-east-1
```

### 2. 配置模型

在同一文件中添加模型配置：

```toml
# Claude 3.5 Sonnet (Bedrock 跨区推理配置文件)
[[models]]
model_identifier = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
name = "claude-3.5-sonnet-bedrock"
api_provider = "bedrock_us_east"
price_in = 3.0   # 每百万输入 token 价格（USD）
price_out = 15.0  # 每百万输出 token 价格（USD）
force_stream_mode = false

# Amazon Nova Pro
[[models]]
model_identifier = "us.amazon.nova-pro-v1:0"
name = "nova-pro"
api_provider = "bedrock_us_east"
price_in = 0.8
price_out = 3.2
force_stream_mode = false

# Llama 3.1 405B
[[models]]
model_identifier = "us.meta.llama3-2-90b-instruct-v1:0"
name = "llama-3.1-405b-bedrock"
api_provider = "bedrock_us_east"
price_in = 0.00532
price_out = 0.016
force_stream_mode = false
```

## 支持的功能

### ✅ 已实现

- **对话生成**：支持多轮对话，自动处理 system prompt
- **流式输出**：支持流式响应（`force_stream_mode = true`）
- **工具调用**：完整支持 Tool Use（函数调用）
- **多模态**：支持图片输入（PNG、JPEG、GIF、WebP）
- **文本嵌入**：支持 Titan Embeddings 等嵌入模型
- **跨区推理**：支持 Inference Profile（如 `us.anthropic.claude-3-5-sonnet-20240620-v1:0`）

### ⚠️ 限制

- **音频转录**：Bedrock 不直接支持语音转文字，建议使用 AWS Transcribe
- **System 角色**：Bedrock Converse API 将 system 消息单独处理，不计入 messages 列表
- **Tool 角色**：暂不支持 Tool 消息回传（需要用 User 角色模拟）

## 模型 ID 参考

### 推理配置文件（跨区）

| 模型 | Model ID | 区域覆盖 |
|------|----------|----------|
| Claude 3.5 Sonnet | `us.anthropic.claude-3-5-sonnet-20240620-v1:0` | us-east-1, us-west-2 |
| Claude 3 Opus | `us.anthropic.claude-3-opus-20240229-v1:0` | 多区 |
| Nova Pro | `us.amazon.nova-pro-v1:0` | 多区 |
| Llama 3.1 405B | `us.meta.llama3-2-90b-instruct-v1:0` | 多区 |

### 单区基础模型

| 模型 | Model ID | 区域 |
|------|----------|------|
| Claude 3.5 Sonnet | `anthropic.claude-3-5-sonnet-20240620-v1:0` | 单区 |
| Nova Micro | `amazon.nova-micro-v1:0` | us-east-1 |
| Nova Lite | `amazon.nova-lite-v1:0` | us-east-1 |
| Titan Embeddings G1 | `amazon.titan-embed-text-v1` | 多区 |

完整模型列表：https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html

## 使用示例

### Python 调用示例

```python
from src.llm_models import get_llm_client
from src.llm_models.payload_content.message import MessageBuilder

# 获取客户端
client = get_llm_client("bedrock_us_east")

# 构建消息
builder = MessageBuilder()
builder.add_user_message("你好，请介绍一下 AWS Bedrock")

# 调用模型
response = await client.get_response(
    model_info=get_model_info("claude-3.5-sonnet-bedrock"),
    message_list=[builder.build()],
    max_tokens=1024,
    temperature=0.7
)

print(response.content)
```

### 多模态示例（图片输入）

```python
import base64

builder = MessageBuilder()
builder.add_text_content("这张图片里有什么？")

# 添加图片（支持 JPEG、PNG、GIF、WebP）
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()
    builder.add_image_content("jpeg", image_data)

builder.set_role_user()

response = await client.get_response(
    model_info=get_model_info("claude-3.5-sonnet-bedrock"),
    message_list=[builder.build()],
    max_tokens=1024
)
```

### 工具调用示例

```python
from src.llm_models.payload_content.tool_option import ToolOption, ToolParam, ParamType

# 定义工具
tool = ToolOption(
    name="get_weather",
    description="获取指定城市的天气信息",
    params=[
        ToolParam(
            name="city",
            param_type=ParamType.String,
            description="城市名称",
            required=True
        )
    ]
)

# 调用
response = await client.get_response(
    model_info=get_model_info("claude-3.5-sonnet-bedrock"),
    message_list=messages,
    tool_options=[tool],
    max_tokens=1024
)

# 检查工具调用
if response.tool_calls:
    for call in response.tool_calls:
        print(f"工具: {call.name}, 参数: {call.arguments}")
```

## 权限配置

### IAM 策略示例

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:Converse",
        "bedrock:ConverseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/*",
        "arn:aws:bedrock:*:*:inference-profile/*"
      ]
    }
  ]
}
```

## 费用优化建议

1. **使用推理配置文件（Inference Profile）**：自动路由到低成本区域
2. **启用缓存**：对于重复的 system prompt，Bedrock 支持提示词缓存
3. **批量处理**：嵌入任务可批量调用，减少请求次数
4. **监控用量**：通过 `LLMUsageRecorder` 自动记录 token 消耗和费用

## 故障排查

### 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `AccessDeniedException` | IAM 权限不足 | 检查 IAM 策略是否包含 `bedrock:InvokeModel` |
| `ResourceNotFoundException` | 模型 ID 错误或区域不支持 | 验证 model_identifier 和 region 配置 |
| `ThrottlingException` | 超过配额限制 | 增加 retry_interval 或申请提额 |
| `ValidationException` | 请求参数错误 | 检查 messages 格式和 max_tokens 范围 |

### 调试模式

启用详细日志：

```python
from src.common.logger import get_logger

logger = get_logger("Bedrock客户端")
logger.setLevel("DEBUG")
```

## 依赖安装

```bash
pip install aioboto3 botocore
```

或使用项目的 `requirements.txt`。

## 参考资料

- [AWS Bedrock 官方文档](https://docs.aws.amazon.com/bedrock/)
- [Converse API 参考](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
- [支持的模型列表](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
- [定价计算器](https://aws.amazon.com/bedrock/pricing/)

---

**集成日期**: 2025年12月6日  
**状态**: ✅ 生产就绪
