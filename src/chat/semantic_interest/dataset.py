"""数据集生成与 LLM 标注

从数据库采样消息并使用 LLM 进行兴趣度标注
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.common.logger import get_logger
from src.config.config import global_config

logger = get_logger("semantic_interest.dataset")


class DatasetGenerator:
    """训练数据集生成器
    
    从历史消息中采样并使用 LLM 进行标注
    """

    # 采样消息时的硬上限，避免一次采样过大导致内存/耗时问题
    HARD_MAX_SAMPLES = 2000

    # 标注提示词模板（单条）
    ANNOTATION_PROMPT = """你是一个帮助标注消息兴趣度的专家。你需要根据人格设定判断该消息是否会引起角色的兴趣。

## 人格信息
{persona_info}

## 消息内容
{message_text}

## 标注规则
请判断角色对这条消息的兴趣程度，返回以下之一：
- **-1**: 完全不感兴趣或排斥（话题不相关、违背价值观、无聊重复等）
- **0**: 中立（可以回应但不特别感兴趣）
- **1**: 感兴趣（话题相关、符合兴趣点、能产生深度对话）

只需返回数字 -1、0 或 1，不要其他内容。"""

    # 批量标注提示词模板
    BATCH_ANNOTATION_PROMPT = """你是一个帮助标注消息兴趣度的专家。你需要根据人格设定判断每条消息是否会引起角色的兴趣。

## 人格信息
{persona_info}

## 标注规则
对每条消息判断角色的兴趣程度：
- **-1**: 完全不感兴趣或排斥（话题不相关、违背价值观、无聊重复等）
- **0**: 中立（可以回应但不特别感兴趣）
- **1**: 感兴趣（话题相关、符合兴趣点、能产生深度对话）

## 消息列表
{messages_list}

## 输出格式
请严格按照以下JSON格式返回，每条消息一个标签：
```json
{example_output}
```

只返回JSON，不要其他内容。"""

    def __init__(
        self,
        model_name: str | None = None,
        max_samples_per_batch: int = 50,
    ):
        """初始化数据集生成器
        
        Args:
            model_name: LLM 模型名称（None 则使用默认模型）
            max_samples_per_batch: 每批次最大采样数
        """
        self.model_name = model_name
        self.max_samples_per_batch = max_samples_per_batch
        self.model_client = None

    async def initialize(self):
        """初始化 LLM 客户端"""
        try:
            from src.llm_models.utils_model import LLMRequest
            from src.config.config import model_config
            
            # 使用 utilities 模型配置（标注更偏工具型）
            if hasattr(model_config.model_task_config, 'utils'):
                self.model_client = LLMRequest(
                    model_set=model_config.model_task_config.utils,
                    request_type="semantic_annotation"
                )
                logger.info(f"数据集生成器初始化完成，使用 utils 模型")
            else:
                logger.error("未找到 utils 模型配置")
                self.model_client = None
        except ImportError as e:
            logger.warning(f"无法导入 LLM 模块: {e}，标注功能将不可用")
            self.model_client = None
        except Exception as e:
            logger.error(f"LLM 客户端初始化失败: {e}")
            self.model_client = None

    async def sample_messages(
        self,
        days: int = 7,
        min_length: int = 5,
        max_samples: int = 1000,
        priority_ranges: list[tuple[float, float]] | None = None,
    ) -> list[dict[str, Any]]:
        """从数据库采样消息（优化版：减少查询量和内存使用）
        
        Args:
            days: 采样最近 N 天的消息
            min_length: 最小消息长度
            max_samples: 最大采样数量
            priority_ranges: 优先采样的兴趣分范围列表，如 [(0.4, 0.6)]
            
        Returns:
            消息样本列表
        """
        from src.common.database.api.query import QueryBuilder
        from src.common.database.core.models import Messages
        from sqlalchemy import func, or_

        logger.info(f"开始采样消息，时间范围: 最近 {days} 天，目标数量: {max_samples}")

        # 限制采样数量硬上限
        requested_max_samples = max_samples
        if max_samples is None:
            max_samples = self.HARD_MAX_SAMPLES
        else:
            max_samples = int(max_samples)
        if max_samples <= 0:
            logger.warning(f"max_samples={requested_max_samples} 非法，返回空样本")
            return []
        if max_samples > self.HARD_MAX_SAMPLES:
            logger.warning(
                f"max_samples={requested_max_samples} 超过硬上限 {self.HARD_MAX_SAMPLES}，"
                f"已截断为 {self.HARD_MAX_SAMPLES}"
            )
            max_samples = self.HARD_MAX_SAMPLES

        # 查询条件
        cutoff_time = datetime.now() - timedelta(days=days)
        cutoff_ts = cutoff_time.timestamp()
        
        # 优化策略：为了过滤掉长度不足的消息，预取 max_samples * 1.5 条
        # 这样可以在保证足够样本的同时减少查询量
        prefetch_limit = int(max_samples * 1.5)
        
        # 构建优化查询：在数据库层面限制数量并按时间倒序（最新消息优先）
        query_builder = QueryBuilder(Messages)
        
        # 过滤条件：时间范围 + 消息文本不为空
        messages = await query_builder.filter(
            time__gte=cutoff_ts,
        ).order_by(
            "-time"  # 按时间倒序，优先采样最新消息
        ).limit(
            prefetch_limit  # 限制预取数量
        ).all(as_dict=True)

        logger.info(f"预取 {len(messages)} 条消息（限制: {prefetch_limit}）")

        # 过滤消息长度和提取文本
        filtered = []
        for msg in messages:
            text = msg.get("processed_plain_text") or msg.get("display_message") or ""
            text = text.strip()
            if text and len(text) >= min_length:
                filtered.append({**msg, "message_text": text})
                # 达到目标数量即可停止
                if len(filtered) >= max_samples:
                    break

        logger.info(f"过滤后得到 {len(filtered)} 条有效消息（目标: {max_samples}）")

        # 如果过滤后数量不足，记录警告
        if len(filtered) < max_samples:
            logger.warning(
                f"过滤后消息数量 ({len(filtered)}) 少于目标 ({max_samples})，"
                f"可能需要扩大采样范围（增加 days 参数或降低 min_length）"
            )

        # 随机打乱样本顺序（避免时间偏向）
        if len(filtered) > 0:
            random.shuffle(filtered)

        # 转换为标准格式
        result = []
        for msg in filtered:
            result.append({
                "message_id": msg.get("message_id"),
                "user_id": msg.get("user_id"),
                "chat_id": msg.get("chat_id"),
                "message_text": msg.get("message_text", ""),
                "timestamp": msg.get("time"),
                "platform": msg.get("chat_info_platform"),
            })

        logger.info(f"采样完成，共 {len(result)} 条消息")
        return result

    async def annotate_message(
        self,
        message_text: str,
        persona_info: dict[str, Any],
    ) -> int:
        """使用 LLM 标注单条消息
        
        Args:
            message_text: 消息文本
            persona_info: 人格信息
            
        Returns:
            标签 (-1, 0, 1)
        """
        if not self.model_client:
            await self.initialize()

        # 构造人格描述
        persona_desc = self._format_persona_info(persona_info)

        # 构造提示词
        prompt = self.ANNOTATION_PROMPT.format(
            persona_info=persona_desc,
            message_text=message_text,
        )

        try:
            if not self.model_client:
                logger.warning("LLM 客户端未初始化，返回默认标签")
                return 0

            # 调用 LLM
            response = await self.model_client.generate_response_async(
                prompt=prompt,
                max_tokens=10,
                temperature=0.1,  # 低温度保证一致性
            )

            # 解析响应
            label = self._parse_label(response)
            return label

        except Exception as e:
            logger.error(f"LLM 标注失败: {e}")
            return 0  # 默认返回中立

    async def annotate_batch(
        self,
        messages: list[dict[str, Any]],
        persona_info: dict[str, Any],
        save_path: Path | None = None,
        batch_size: int = 20,
    ) -> list[dict[str, Any]]:
        """批量标注消息（真正的批量模式）
        
        Args:
            messages: 消息列表
            persona_info: 人格信息
            save_path: 保存路径（可选）
            batch_size: 每次LLM请求处理的消息数（默认20）
            
        Returns:
            标注后的数据集
        """
        logger.info(f"开始批量标注，共 {len(messages)} 条消息，每批 {batch_size} 条")

        annotated_data = []

        for i in range(0, len(messages), batch_size):
            batch = messages[i : i + batch_size]
            
            # 批量标注（一次LLM请求处理多条消息）
            labels = await self._annotate_batch_llm(batch, persona_info)
            
            # 保存结果
            for msg, label in zip(batch, labels):
                annotated_data.append({
                    "message_id": msg["message_id"],
                    "message_text": msg["message_text"],
                    "label": label,
                    "user_id": msg.get("user_id"),
                    "chat_id": msg.get("chat_id"),
                    "timestamp": msg.get("timestamp"),
                })

            logger.info(f"已标注 {len(annotated_data)}/{len(messages)} 条")

        # 统计标签分布
        label_counts = {}
        for item in annotated_data:
            label = item["label"]
            label_counts[label] = label_counts.get(label, 0) + 1

        logger.info(f"标注完成，标签分布: {label_counts}")

        # 保存到文件
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(annotated_data, f, ensure_ascii=False, indent=2)
            logger.info(f"数据集已保存到: {save_path}")

        return annotated_data

    async def _annotate_batch_llm(
        self,
        messages: list[dict[str, Any]],
        persona_info: dict[str, Any],
    ) -> list[int]:
        """使用一次LLM请求标注多条消息
        
        Args:
            messages: 消息列表（通常20条）
            persona_info: 人格信息
            
        Returns:
            标签列表
        """
        if not self.model_client:
            logger.warning("LLM 客户端未初始化，返回默认标签")
            return [0] * len(messages)

        # 构造人格描述
        persona_desc = self._format_persona_info(persona_info)

        # 构造消息列表
        messages_list = ""
        for idx, msg in enumerate(messages, 1):
            messages_list += f"{idx}. {msg['message_text']}\n"

        # 构造示例输出
        example_output = json.dumps(
            {str(i): 0 for i in range(1, len(messages) + 1)},
            ensure_ascii=False,
            indent=2
        )

        # 构造提示词
        prompt = self.BATCH_ANNOTATION_PROMPT.format(
            persona_info=persona_desc,
            messages_list=messages_list,
            example_output=example_output,
        )

        try:
            # 调用 LLM（使用更大的token限制）
            response = await self.model_client.generate_response_async(
                prompt=prompt,
                max_tokens=500,  # 批量标注需要更多token
                temperature=0.1,
            )

            # 解析批量响应
            labels = self._parse_batch_labels(response, len(messages))
            return labels

        except Exception as e:
            logger.error(f"批量LLM标注失败: {e}，返回默认值")
            return [0] * len(messages)

    def _format_persona_info(self, persona_info: dict[str, Any]) -> str:
        """格式化人格信息
        
        Args:
            persona_info: 人格信息字典
            
        Returns:
            格式化后的人格描述
        """
        def _stringify(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, (list, tuple, set)):
                return "、".join([str(v) for v in value if v is not None and str(v).strip()])
            if isinstance(value, dict):
                try:
                    return json.dumps(value, ensure_ascii=False, sort_keys=True)
                except Exception:
                    return str(value)
            return str(value).strip()

        parts: list[str] = []

        name = _stringify(persona_info.get("name"))
        if name:
            parts.append(f"角色名称: {name}")

        # 核心/侧面/身份等完整人设信息
        personality_core = _stringify(persona_info.get("personality_core"))
        if personality_core:
            parts.append(f"核心人设: {personality_core}")

        personality_side = _stringify(persona_info.get("personality_side"))
        if personality_side:
            parts.append(f"侧面特质: {personality_side}")

        identity = _stringify(persona_info.get("identity"))
        if identity:
            parts.append(f"身份特征: {identity}")

        # 追加其他未覆盖字段（保持信息完整）
        known_keys = {
            "name",
            "personality_core",
            "personality_side",
            "identity",
        }
        for key, value in persona_info.items():
            if key in known_keys:
                continue
            value_str = _stringify(value)
            if value_str:
                parts.append(f"{key}: {value_str}")

        return "\n".join(parts) if parts else "无特定人格设定"

    def _parse_label(self, response: str) -> int:
        """解析 LLM 响应为标签
        
        Args:
            response: LLM 响应文本
            
        Returns:
            标签 (-1, 0, 1)
        """
        # 部分 LLM 客户端可能返回 (text, meta) 的 tuple，这里取首元素并转为字符串
        if isinstance(response, (tuple, list)):
            response = response[0] if response else ""
        response = str(response).strip()

        # 尝试直接解析数字
        if response in ["-1", "0", "1"]:
            return int(response)

        # 尝试提取数字
        if "-1" in response:
            return -1
        elif "1" in response:
            return 1
        elif "0" in response:
            return 0

        # 默认返回中立
        logger.warning(f"无法解析 LLM 响应: {response}，返回默认值 0")
        return 0

    def _parse_batch_labels(self, response: str, expected_count: int) -> list[int]:
        """解析批量LLM响应为标签列表
        
        Args:
            response: LLM 响应文本（JSON格式）
            expected_count: 期望的标签数量
            
        Returns:
            标签列表
        """
        try:
            # 兼容 tuple/list 返回格式
            if isinstance(response, (tuple, list)):
                response = response[0] if response else ""
            response = str(response)

            # 提取JSON内容
            import re
            json_match = re.search(r'```json\s*({.*?})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析
                json_str = response
            import json_repair
            # 解析JSON
            labels_json = json_repair.repair_json(json_str)
            labels_dict = json.loads(labels_json)  # 验证是否为有效JSON
            # 转换为列表
            labels = []
            for i in range(1, expected_count + 1):
                key = str(i)
                if key in labels_dict:
                    label = labels_dict[key]
                    # 确保标签值有效
                    if label in [-1, 0, 1]:
                        labels.append(label)
                    else:
                        logger.warning(f"无效标签值 {label}，使用默认值 0")
                        labels.append(0)
                else:
                    # 尝试从值列表或数组中顺序取值
                    if isinstance(labels_dict, list) and len(labels_dict) >= i:
                        label = labels_dict[i - 1]
                        labels.append(label if label in [-1, 0, 1] else 0)
                    else:
                        labels.append(0)

            if len(labels) != expected_count:
                logger.warning(
                    f"标签数量不匹配：期望 {expected_count}，实际 {len(labels)}，"
                    f"补齐为 {expected_count}"
                )
                # 补齐或截断
                if len(labels) < expected_count:
                    labels.extend([0] * (expected_count - len(labels)))
                else:
                    labels = labels[:expected_count]

            return labels

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}，响应内容: {response[:200]}")
            return [0] * expected_count
        except Exception as e:
            # 兜底：尝试直接提取所有标签数字
            try:
                import re
                numbers = re.findall(r"-?1|0", response)
                labels = [int(n) for n in numbers[:expected_count]]
                if len(labels) < expected_count:
                    labels.extend([0] * (expected_count - len(labels)))
                return labels
            except Exception:
                logger.error(f"批量标签解析失败: {e}")
                return [0] * expected_count

    @staticmethod
    def load_dataset(path: Path) -> tuple[list[str], list[int]]:
        """加载训练数据集
        
        Args:
            path: 数据集文件路径
            
        Returns:
            (文本列表, 标签列表)
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = [item["message_text"] for item in data]
        labels = [item["label"] for item in data]

        logger.info(f"加载数据集: {len(texts)} 条样本")
        return texts, labels


async def generate_training_dataset(
    output_path: Path,
    persona_info: dict[str, Any],
    days: int = 7,
    max_samples: int = 1000,
    model_name: str | None = None,
) -> Path:
    """生成训练数据集（主函数）
    
    Args:
        output_path: 输出文件路径
        persona_info: 人格信息
        days: 采样最近 N 天的消息
        max_samples: 最大采样数
        model_name: LLM 模型名称
        
    Returns:
        保存的文件路径
    """
    generator = DatasetGenerator(model_name=model_name)
    await generator.initialize()

    # 采样消息
    messages = await generator.sample_messages(
        days=days,
        max_samples=max_samples,
    )

    # 批量标注
    await generator.annotate_batch(
        messages=messages,
        persona_info=persona_info,
        save_path=output_path,
    )

    return output_path
