"""
Music Plugin - 网易云音乐点歌插件

基于网易云音乐API的智能点歌插件，支持音乐搜索和点歌功能。

功能特性：
- 智能音乐搜索和推荐
- 支持关键词自动触发和命令手动触发
- 丰富的音乐信息展示
- 专辑封面显示
- 灵活的配置选项

使用方法：
- Action触发：发送包含"音乐"、"歌曲"等关键词的消息
- Command触发：/music 歌曲名

API接口：https://api.vkeys.cn/v2/music/netease
"""

from typing import List, Tuple, Type, Optional
import aiohttp
import json
import requests
import base64
import asyncio  # 新增
from src.plugin_system.apis import send_api, chat_api, database_api, generator_api
from src.plugin_system import (
    BasePlugin, register_plugin, BaseAction, BaseCommand,
    ComponentInfo, ActionActivationType, ChatMode
)
from src.plugin_system.base.config_types import ConfigField
from src.common.logger import get_logger

logger = get_logger("music_plugin")

# ===== 智能消息发送工具 =====
async def smart_send(chat_stream, message_data):
    """智能发送不同类型的消息，并返回实际发包内容"""
    message_type = message_data.get("type", "text")
    content = message_data.get("content", "")
    options = message_data.get("options", {})
    target_id = (chat_stream.group_info.group_id if getattr(chat_stream, 'group_info', None)
                else chat_stream.user_info.user_id)
    is_group = getattr(chat_stream, 'group_info', None) is not None
    # 调试用，记录实际发包内容
    packet = {
        "message_type": message_type,
        "content": content,
        "target_id": target_id,
        "is_group": is_group,
        "typing": options.get("typing", False),
        "reply_to": options.get("reply_to", ""),
        "display_message": options.get("display_message", "")
    }
    print(f"[调试] smart_send 发包内容: {json.dumps(packet, ensure_ascii=False)}")
    # 实际发送
    success = await send_api.custom_message(
        message_type=message_type,
        content=content,
        target_id=target_id,
        is_group=is_group,
        typing=options.get("typing", False),
        reply_to=options.get("reply_to", ""),
        display_message=options.get("display_message", "")
    )
    return success, packet

# ===== Action组件 =====

class MusicSearchAction(BaseAction):
    """音乐搜索Action - 智能音乐推荐"""

    action_name = "music_search"
    action_description = "搜索并推荐音乐"

    # 关键词或LLM混合激活
    focus_activation_type = ActionActivationType.KEYWORD_OR_LLM_JUDGE
    normal_activation_type = ActionActivationType.KEYWORD_OR_LLM_JUDGE
    activation_keywords = ["音乐", "歌曲", "点歌", "听歌", "music", "song", "播放", "来首"]

    action_parameters = {
        "song_name": "要搜索的歌曲名称"
    }
    action_require = [
       "当用户想要听音乐、点歌、或询问音乐相关信息时使用。",
       "这是一个纯粹的音乐搜索动作，它只负责找到歌曲并发送卡片。",
       "回复和交互逻辑应由上层 Planner 决定，可以将此动作与'reply'动作组合使用，以实现更拟人化的交互。"
    ]
    associated_types = ["text"]

    def get_log_prefix(self) -> str:
        """获取日志前缀"""
        return f"[MusicSearchAction]"

    async def execute(self) -> Tuple[bool, str]:
        """执行音乐搜索"""
        try:
            # 获取参数

            song_name = self.action_data.get("song_name", "").strip()
            if not song_name:
                await self._send_dynamic_reply(
                    raw_reply="[缺少歌曲名称]",
                    reason="用户没有提供歌曲名称",
                    emotion="疑惑"
                )
                return False, "缺少歌曲名称"

            # 从配置获取设置
            api_url = self.get_config("api.base_url", "https://api.vkeys.cn")
            timeout = self.get_config("api.timeout", 10)

            logger.info(f"{self.get_log_prefix()} 开始搜索音乐，歌曲：{song_name[:50]}...")

            # 调用音乐API
            music_info = await self._call_music_api(api_url, song_name, timeout)

            if music_info:
                # 发送音乐信息
                await self._send_music_info(music_info)

                # 记录动作信息
                song_name_display = music_info.get('song', '未知歌曲')
                singer_display = music_info.get('singer', '未知歌手')
                await self.store_action_info(
                    action_build_into_prompt=True,
                    action_prompt_display=f"为用户搜索并推荐了音乐：{song_name_display} - {singer_display}",
                    action_done=True
                )

                logger.info(f"{self.get_log_prefix()} 音乐搜索成功")
                return True, f"成功找到音乐：{music_info.get('song', '未知')[:30]}..."
            else:
                await self._send_dynamic_reply(
                    raw_reply="[未找到音乐]",
                    reason=f"API未能根据关键词 '{song_name}' 找到任何音乐",
                    emotion="遗憾",
                    context={"song_name": song_name}
                )
                return False, "未找到音乐"

        except Exception as e:
            logger.error(f"{self.get_log_prefix()} 音乐搜索出错: {e}")
            await self._send_dynamic_reply(
                raw_reply="[API请求异常]",
                reason=f"调用音乐API时发生异常: {e}",
                emotion="抱歉",
                context={"error": str(e)}
            )
            return False, f"音乐搜索出错: {e}"

    async def _call_music_api(self, api_url: str, song_name: str, timeout: int, retries: int = 3, delay: float = 1.5) -> Optional[dict]:
        """调用音乐API搜索歌曲，带重试机制"""
        for attempt in range(1, retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                    params = {
                        "word": song_name,
                        "choose": 1  # 选择第一首
                    }

                    async with session.get(f"{api_url}/v2/music/netease", params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("code") == 200:
                                return data.get("data", {})
                            else:
                                logger.warning(f"{self.get_log_prefix()} API返回错误: {data.get('message', '未知错误')}")
                        else:
                            logger.warning(f"{self.get_log_prefix()} API请求失败，状态码: {response.status}")
            except Exception as e:
                logger.error(f"{self.get_log_prefix()} 第{attempt}次调用音乐API出错: {e}")
            if attempt < retries:
                await asyncio.sleep(delay)
        return None

    async def _send_music_info(self, music_info: dict):
        """发送音乐信息"""
        try:
            song_id = music_info.get("id", "")

            # 根据配置决定是否发送详细信息
            if self.get_config("features.show_detailed_info", False):
                song = music_info.get("song", "未知歌曲")
                singer = music_info.get("singer", "未知歌手")
                album = music_info.get("album", "未知专辑")
                interval = music_info.get("interval", "未知时长")
                message = f"🎵 歌曲：{song}\n"
                message += f"👤 歌手：{singer}\n"
                message += f"💿 专辑：{album}\n"
                message += f"⏱️ 时长：{interval}\n"
                await self.send_text(message)

            # 发送音乐卡片
            if song_id:
                await self.send_custom(message_type="music", content=song_id)
                logger.info(f"{self.get_log_prefix()} 发送音乐卡片成功，ID: {song_id}")
            else:
                logger.warning(f"{self.get_log_prefix()} 音乐ID为空，无法发送音乐卡片")

        except Exception as e:
            logger.error(f"{self.get_log_prefix()} 发送音乐信息出错: {e}")
            await self.send_text("❌ 发送音乐信息时出现错误")

    async def _send_dynamic_reply(self, raw_reply: str, reason: str, emotion: str, context: dict = None):
        """使用生成器API发送动态回复"""
        try:
            reply_data = {
                "raw_reply": raw_reply,
                "reason": reason,
                "emotion": emotion,
                "context": context or {}
            }
            success, reply_set, _ = await generator_api.generate_reply(
                chat_stream=self.chat_stream,
                action_data=reply_data,
                enable_splitter=True,
                enable_chinese_typo=True
            )
            if success and reply_set:
                for reply_type, reply_content in reply_set:
                    if reply_type == "text":
                        await self.send_text(reply_content)
        except Exception as e:
            logger.error(f"发送动态回复时出错: {e}")

# ===== Command组件 =====
class MusicCommand(BaseCommand):
    """音乐点歌Command - 直接点歌命令"""

    command_name = "music"
    command_description = "点歌命令"
    command_pattern = r"^/music\s+(?P<song_name>.+)$"  # 用命名组
    command_help = "点歌命令，用法：/music 歌曲名"
    command_examples = ["/music 勾指起誓", "/music 晴天", "/music Jay Chou 青花瓷"]
    intercept_message = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 从kwargs中获取chat_stream并赋值给实例变量
        self.chat_stream = kwargs.get('chat_stream')

    def get_log_prefix(self) -> str:
        """获取日志前缀"""
        return f"[MusicCommand]"

    async def _send_dynamic_reply(self, raw_reply: str, reason: str, emotion: str, context: dict = None):
        """使用生成器API发送动态回复"""
        try:
            reply_data = {
                "raw_reply": raw_reply,
                "reason": reason,
                "emotion": emotion,
                "context": context or {}
            }
            success, reply_set, _ = await generator_api.generate_reply(
                chat_stream=self.chat_stream,
                action_data=reply_data,
                enable_splitter=True,
                enable_chinese_typo=True
            )
            if success and reply_set:
                for reply_type, reply_content in reply_set:
                    if reply_type == "text":
                        await self.send_text(reply_content)
        except Exception as e:
            logger.error(f"发送动态回复时出错: {e}")

    async def execute(self) -> Tuple[bool, str, bool]:
        """执行音乐点歌命令"""
        try:
            # 获取匹配的参数
            song_name = (self.matched_groups or {}).get("song_name", "").strip()

            if not song_name:
                await self.send_text("❌ 请输入正确的格式：/music 歌曲名")
                return False, "缺少歌曲名称", True

            # 从配置获取设置
            api_url = self.get_config("api.base_url", "https://api.vkeys.cn")
            timeout = self.get_config("api.timeout", 10)

            logger.info(f"{self.get_log_prefix()} 执行点歌命令，歌曲：{song_name[:50]}...")

            # 调用音乐API
            music_info = await self._call_music_api(api_url, song_name, timeout)

            if music_info:
                # 发送音乐信息
                await self._send_detailed_music_info(music_info)

                logger.info(f"{self.get_log_prefix()} 点歌成功")
                return True, f"成功点歌：{music_info.get('song', '未知')[:30]}...", True
            else:
                await self._send_dynamic_reply(
                    raw_reply="[未找到音乐]",
                    reason=f"API未能根据关键词 '{song_name}' 找到任何音乐",
                    emotion="遗憾",
                    context={"song_name": song_name}
                )
                return False, "未找到音乐", True

        except Exception as e:
            logger.error(f"{self.get_log_prefix()} 点歌命令执行出错: {e}")
            await self._send_dynamic_reply(
                raw_reply="[API请求异常]",
                reason=f"调用音乐API时发生异常: {e}",
                emotion="抱歉",
                context={"error": str(e)}
            )
            return False, f"点歌失败: {e}", True

    async def _call_music_api(self, api_url: str, song_name: str, timeout: int, retries: int = 3, delay: float = 1.5) -> Optional[dict]:
        """调用音乐API搜索歌曲，带重试机制"""
        for attempt in range(1, retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                    params = {
                        "word": song_name,
                        "choose": 1  # 选择第一首
                    }

                    async with session.get(f"{api_url}/v2/music/netease", params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("code") == 200:
                                return data.get("data", {})
                            else:
                                logger.warning(f"{self.get_log_prefix()} API返回错误: {data.get('message', '未知错误')}")
                        else:
                            logger.warning(f"{self.get_log_prefix()} API请求失败，状态码: {response.status}")
            except Exception as e:
                logger.error(f"{self.get_log_prefix()} 第{attempt}次调用音乐API出错: {e}")
            if attempt < retries:
                await asyncio.sleep(delay)
        return None

    async def _send_detailed_music_info(self, music_info: dict):
        """发送详细音乐信息（Command用）"""
        try:
            song_id = music_info.get("id", "")

            # 根据配置决定是否发送详细信息
            if self.get_config("features.show_detailed_info", False):
                song = music_info.get("song", "未知歌曲")
                singer = music_info.get("singer", "未知歌手")
                album = music_info.get("album", "未知专辑")
                interval = music_info.get("interval", "未知时长")
                message = f"🎵 歌曲：{song}\n"
                message += f"👤 歌手：{singer}\n"
                message += f"💿 专辑：{album}\n"
                message += f"⏱️ 时长：{interval}\n"
                await self.send_text(message)

            # 发送音乐卡片
            if song_id:
                await self.send_type(message_type="music", content=song_id)
                logger.info(f"{self.get_log_prefix()} 发送音乐卡片成功，ID: {song_id}")
            else:
                logger.warning(f"{self.get_log_prefix()} 音乐ID为空，无法发送音乐卡片")

        except Exception as e:
            logger.error(f"{self.get_log_prefix()} 发送详细音乐信息出错: {e}")
            await self.send_text("❌ 发送音乐信息时出现错误")
# ===== 插件注册 =====

@register_plugin
class MusicPlugin(BasePlugin):
    """音乐点歌插件 - 基于网易云音乐API的智能点歌插件"""

    plugin_name = "music_plugin"
    plugin_description = "网易云音乐点歌插件，支持音乐搜索和点歌功能"
    plugin_version = "1.0.0"
    plugin_author = "Augment Agent"
    enable_plugin = True
    config_file_name = "config.toml"
    dependencies = []  # 插件依赖列表
    python_dependencies = ["aiohttp", "requests"]  # Python包依赖列表

    # 配置节描述
    config_section_descriptions = {
        "plugin": "插件基本配置",
        "components": "组件启用控制",
        "api": "API接口配置",
        "music": "音乐功能配置",
        "features": "功能开关配置"
    }

    # 配置Schema
    config_schema = {
        "plugin": {
            "enabled": ConfigField(type=bool, default=True, description="是否启用插件")
        },
        "components": {
            "action_enabled": ConfigField(type=bool, default=True, description="是否启用Action组件"),
            "command_enabled": ConfigField(type=bool, default=True, description="是否启用Command组件")
        },
        "api": {
            "base_url": ConfigField(
                type=str,
                default="https://api.vkeys.cn",
                description="音乐API基础URL"
            ),
            "timeout": ConfigField(type=int, default=10, description="API请求超时时间(秒)")
        },
        "music": {
            "default_quality": ConfigField(
                type=str,
                default="9",
                description="默认音质等级(1-9)"
            ),
            "max_search_results": ConfigField(
                type=int,
                default=10,
                description="最大搜索结果数"
            )
        },
        "features": {
            "show_cover": ConfigField(type=bool, default=True, description="是否显示专辑封面"),
            "show_download_link": ConfigField(
                type=bool,
                default=False,
                description="是否显示下载链接"
            ),
            "show_detailed_info": ConfigField(type=bool, default=False, description="是否显示详细的音乐信息文本"),
            "send_as_voice": ConfigField(type=bool, default=False, description="是否以语音消息发送音乐（true=语音消息，false=音乐卡片）")
        }
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        """返回插件组件列表"""
        components = []

        # 根据配置决定是否启用组件
        if self.get_config("components.action_enabled", True):
            components.append((MusicSearchAction.get_action_info(), MusicSearchAction))

        if self.get_config("components.command_enabled", True):
            components.append((MusicCommand.get_command_info(), MusicCommand))

        return components
