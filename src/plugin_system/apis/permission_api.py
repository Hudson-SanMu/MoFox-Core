"""
权限系统API - 提供权限管理相关的API接口

这个模块提供了权限系统的核心API，包括权限检查、权限节点管理等功能。
插件可以通过这些API来检查用户权限和管理权限节点。
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.common.logger import get_logger

logger = get_logger(__name__)


class PermissionLevel(Enum):
    """权限等级枚举"""

    MASTER = "master"  # 最高权限，无视所有权限节点


@dataclass
class PermissionNode:
    """权限节点数据类"""

    node_name: str  # 权限节点名称，如 "plugin.example.command.test"
    description: str  # 权限节点描述
    plugin_name: str  # 所属插件名称
    default_granted: bool = False  # 默认是否授权


@dataclass
class UserInfo:
    """用户信息数据类"""

    platform: str  # 平台类型，如 "qq"
    user_id: str  # 用户ID

    def __post_init__(self):
        """确保user_id是字符串类型"""
        self.user_id = str(self.user_id)

    def to_tuple(self) -> tuple[str, str]:
        """转换为元组格式"""
        return self.platform, self.user_id


class IPermissionManager(ABC):
    """权限管理器接口"""

    @abstractmethod
    def check_permission(self, user: UserInfo, permission_node: str) -> bool:
        """
        检查用户是否拥有指定权限节点

        Args:
            user: 用户信息
            permission_node: 权限节点名称

        Returns:
            bool: 是否拥有权限
        """
        pass

    @abstractmethod
    def is_master(self, user: UserInfo) -> bool:
        """
        检查用户是否为Master用户

        Args:
            user: 用户信息

        Returns:
            bool: 是否为Master用户
        """
        pass

    @abstractmethod
    def register_permission_node(self, node: PermissionNode) -> bool:
        """
        注册权限节点

        Args:
            node: 权限节点

        Returns:
            bool: 注册是否成功
        """
        pass

    @abstractmethod
    def grant_permission(self, user: UserInfo, permission_node: str) -> bool:
        """
        授权用户权限节点

        Args:
            user: 用户信息
            permission_node: 权限节点名称

        Returns:
            bool: 授权是否成功
        """
        pass

    @abstractmethod
    def revoke_permission(self, user: UserInfo, permission_node: str) -> bool:
        """
        撤销用户权限节点

        Args:
            user: 用户信息
            permission_node: 权限节点名称

        Returns:
            bool: 撤销是否成功
        """
        pass

    @abstractmethod
    def get_user_permissions(self, user: UserInfo) -> List[str]:
        """
        获取用户拥有的所有权限节点

        Args:
            user: 用户信息

        Returns:
            List[str]: 权限节点列表
        """
        pass

    @abstractmethod
    def get_all_permission_nodes(self) -> List[PermissionNode]:
        """
        获取所有已注册的权限节点

        Returns:
            List[PermissionNode]: 权限节点列表
        """
        pass

    @abstractmethod
    def get_plugin_permission_nodes(self, plugin_name: str) -> List[PermissionNode]:
        """
        获取指定插件的所有权限节点

        Args:
            plugin_name: 插件名称

        Returns:
            List[PermissionNode]: 权限节点列表
        """
        pass


class PermissionAPI:
    """权限系统API类"""

    def __init__(self):
        self._permission_manager: Optional[IPermissionManager] = None

    def set_permission_manager(self, manager: IPermissionManager):
        """设置权限管理器实例"""
        self._permission_manager = manager
        logger.info("权限管理器已设置")

    def _ensure_manager(self):
        """确保权限管理器已设置"""
        if self._permission_manager is None:
            raise RuntimeError("权限管理器未设置，请先调用 set_permission_manager")

    def check_permission(self, platform: str, user_id: str, permission_node: str) -> bool:
        """
        检查用户是否拥有指定权限节点

        Args:
            platform: 平台类型，如 "qq"
            user_id: 用户ID
            permission_node: 权限节点名称

        Returns:
            bool: 是否拥有权限

        Raises:
            RuntimeError: 权限管理器未设置时抛出
        """
        self._ensure_manager()
        user = UserInfo(platform=platform, user_id=str(user_id))
        return self._permission_manager.check_permission(user, permission_node)

    def is_master(self, platform: str, user_id: str) -> bool:
        """
        检查用户是否为Master用户

        Args:
            platform: 平台类型，如 "qq"
            user_id: 用户ID

        Returns:
            bool: 是否为Master用户

        Raises:
            RuntimeError: 权限管理器未设置时抛出
        """
        self._ensure_manager()
        user = UserInfo(platform=platform, user_id=str(user_id))
        return self._permission_manager.is_master(user)

    def register_permission_node(
        self, node_name: str, description: str, plugin_name: str, default_granted: bool = False
    ) -> bool:
        """
        注册权限节点

        Args:
            node_name: 权限节点名称，如 "plugin.example.command.test"
            description: 权限节点描述
            plugin_name: 所属插件名称
            default_granted: 默认是否授权

        Returns:
            bool: 注册是否成功

        Raises:
            RuntimeError: 权限管理器未设置时抛出
        """
        self._ensure_manager()
        node = PermissionNode(
            node_name=node_name, description=description, plugin_name=plugin_name, default_granted=default_granted
        )
        return self._permission_manager.register_permission_node(node)

    def grant_permission(self, platform: str, user_id: str, permission_node: str) -> bool:
        """
        授权用户权限节点

        Args:
            platform: 平台类型，如 "qq"
            user_id: 用户ID
            permission_node: 权限节点名称

        Returns:
            bool: 授权是否成功

        Raises:
            RuntimeError: 权限管理器未设置时抛出
        """
        self._ensure_manager()
        user = UserInfo(platform=platform, user_id=str(user_id))
        return self._permission_manager.grant_permission(user, permission_node)

    def revoke_permission(self, platform: str, user_id: str, permission_node: str) -> bool:
        """
        撤销用户权限节点

        Args:
            platform: 平台类型，如 "qq"
            user_id: 用户ID
            permission_node: 权限节点名称

        Returns:
            bool: 撤销是否成功

        Raises:
            RuntimeError: 权限管理器未设置时抛出
        """
        self._ensure_manager()
        user = UserInfo(platform=platform, user_id=str(user_id))
        return self._permission_manager.revoke_permission(user, permission_node)

    def get_user_permissions(self, platform: str, user_id: str) -> List[str]:
        """
        获取用户拥有的所有权限节点

        Args:
            platform: 平台类型，如 "qq"
            user_id: 用户ID

        Returns:
            List[str]: 权限节点列表

        Raises:
            RuntimeError: 权限管理器未设置时抛出
        """
        self._ensure_manager()
        user = UserInfo(platform=platform, user_id=str(user_id))
        return self._permission_manager.get_user_permissions(user)

    def get_all_permission_nodes(self) -> List[Dict[str, Any]]:
        """
        获取所有已注册的权限节点

        Returns:
            List[Dict[str, Any]]: 权限节点列表，每个节点包含 node_name, description, plugin_name, default_granted

        Raises:
            RuntimeError: 权限管理器未设置时抛出
        """
        self._ensure_manager()
        nodes = self._permission_manager.get_all_permission_nodes()
        return [
            {
                "node_name": node.node_name,
                "description": node.description,
                "plugin_name": node.plugin_name,
                "default_granted": node.default_granted,
            }
            for node in nodes
        ]

    def get_plugin_permission_nodes(self, plugin_name: str) -> List[Dict[str, Any]]:
        """
        获取指定插件的所有权限节点

        Args:
            plugin_name: 插件名称

        Returns:
            List[Dict[str, Any]]: 权限节点列表

        Raises:
            RuntimeError: 权限管理器未设置时抛出
        """
        self._ensure_manager()
        nodes = self._permission_manager.get_plugin_permission_nodes(plugin_name)
        return [
            {
                "node_name": node.node_name,
                "description": node.description,
                "plugin_name": node.plugin_name,
                "default_granted": node.default_granted,
            }
            for node in nodes
        ]


# 全局权限API实例
permission_api = PermissionAPI()
