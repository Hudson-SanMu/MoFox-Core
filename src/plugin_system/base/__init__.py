"""
插件基础类模块

提供插件开发的基础类和类型定义
"""

from .base_plugin import BasePlugin
from .base_action import BaseAction
from .base_tool import BaseTool
from .base_command import BaseCommand
from .base_events_handler import BaseEventHandler
from .component_types import (
    ComponentType,
    ActionActivationType,
    ChatMode,
    ChatType,
    ComponentInfo,
    ActionInfo,
    CommandInfo,
    PlusCommandInfo,
    ToolInfo,
    PluginInfo,
    PythonDependency,
    EventHandlerInfo,
    EventType,
    MaiMessages,
    ToolParamType,
)
from .config_types import ConfigField
from .plus_command import PlusCommand, PlusCommandAdapter, create_plus_command_adapter
from .command_args import CommandArgs

__all__ = [
    "BasePlugin",
    "BaseAction",
    "BaseCommand",
    "BaseTool",
    "ComponentType",
    "ActionActivationType",
    "ChatMode",
    "ChatType",
    "ComponentInfo",
    "ActionInfo",
    "CommandInfo",
    "PlusCommandInfo",
    "ToolInfo",
    "PluginInfo",
    "PythonDependency",
    "ConfigField",
    "EventHandlerInfo",
    "EventType",
    "BaseEventHandler",
    "MaiMessages",
    "ToolParamType",
    # 增强命令系统
    "PlusCommand",
    "CommandArgs",
    "PlusCommandAdapter",
    "create_plus_command_adapter",
]
