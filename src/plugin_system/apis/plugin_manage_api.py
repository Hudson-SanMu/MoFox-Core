"""
Plugin Manage API
=================

该模块提供了用于管理插件生命周期的核心API。
功能包括插件的加载、重载、注册、扫描，以及插件的启用/禁用和卸载。

主要功能包括：
- 插件生命周期管理（加载、重载、注册、发现）
- 插件的启用/禁用
- 插件卸载

组件状态管理相关功能请使用 component_state_api
信息查询和报告相关功能请使用 plugin_info_api
"""

import os

from src.common.logger import get_logger
from src.plugin_system.core.plugin_manager import plugin_manager

# 初始化日志记录器
logger = get_logger("plugin_manage_api")


# --------------------------------------------------------------------------------
# Section 1: 插件生命周期管理 (Plugin Lifecycle Management)
# --------------------------------------------------------------------------------
# 该部分包含控制插件加载、重载、注册和发现的核心功能。


async def reload_all_plugins() -> bool:
    """
    重新加载所有当前已成功加载的插件。

    此操作会遍历所有已加载的插件，逐一进行卸载和重新加载。
    如果任何一个插件重载失败，整个过程会继续，但最终返回 False。

    Returns:
        bool: 如果所有插件都成功重载，则为 True，否则为 False。
    """
    logger.info("开始重新加载所有插件...")
    # 使用 list() 创建一个当前已加载插件列表的副本，以避免在迭代过程中修改原始列表
    loaded_plugins = list(plugin_manager.list_loaded_plugins())
    all_success = True

    # 遍历副本列表中的每个插件进行重载
    for plugin_name in loaded_plugins:
        try:
            success = await reload_plugin(plugin_name)
            if not success:
                all_success = False
                logger.error(f"重载插件 {plugin_name} 失败。")
        except Exception as e:
            all_success = False
            logger.error(f"重载插件 {plugin_name} 时发生未知异常: {e}", exc_info=True)

    logger.info("所有插件重载完毕。")
    return all_success


async def reload_plugin(name: str) -> bool:
    """
    重新加载指定的单个插件。

    该函数首先检查插件是否已注册，然后调用插件管理器执行重载操作。

    Args:
        name (str): 要重载的插件的名称。

    Returns:
        bool: 如果插件成功重载，则为 True。

    Raises:
        ValueError: 如果插件未在插件管理器中注册。
    """
    # 验证插件是否存在于注册列表中
    if name not in plugin_manager.list_registered_plugins():
        raise ValueError(f"插件 '{name}' 未注册，无法重载。")
    # 调用插件管理器的核心重载方法
    return await plugin_manager.reload_registered_plugin(name)


def rescan_and_register_plugins(load_after_register: bool = True) -> tuple[int, int]:
    """
    重新扫描所有插件目录，以发现并注册新插件。

    此函数会触发插件管理器扫描其配置的所有插件目录。
    可以选择在注册新发现的插件后立即加载它们。

    Args:
        load_after_register (bool): 如果为 True，新发现的插件将在注册后立即被加载。默认为 True。

    Returns:
        tuple[int, int]: 一个元组，包含 (成功加载的插件数量, 加载失败的插件数量)。
    """
    # 扫描插件目录，获取新注册成功和失败的数量
    success_count, fail_count = plugin_manager.rescan_plugin_directory()

    # 如果不需要在注册后加载，则直接返回扫描结果
    if not load_after_register:
        return success_count, fail_count

    # 找出新注册但尚未加载的插件
    newly_registered = [
        p for p in plugin_manager.list_registered_plugins() if p not in plugin_manager.list_loaded_plugins()
    ]

    loaded_success_count = 0
    # 尝试加载所有新注册的插件
    for plugin_name in newly_registered:
        status, _ = plugin_manager.load_registered_plugin_classes(plugin_name)
        if status:
            loaded_success_count += 1

    # 计算总的成功和失败数量
    total_failed = fail_count + (len(newly_registered) - loaded_success_count)
    return loaded_success_count, total_failed


def register_plugin_from_file(plugin_name: str, load_after_register: bool = True) -> bool:
    """
    从插件目录中查找、注册并选择性地加载一个指定的插件。

    如果插件已经加载，此函数将直接返回 True。
    如果插件未注册，它会遍历所有插件目录以查找匹配的插件文件夹。

    Args:
        plugin_name (str): 插件的名称（通常是其目录名）。
        load_after_register (bool): 注册成功后是否立即加载该插件。默认为 True。

    Returns:
        bool: 如果插件成功注册（并且根据参数成功加载），则为 True。
    """
    # 如果插件已经加载，无需执行任何操作
    if plugin_name in plugin_manager.list_loaded_plugins():
        logger.warning(f"插件 '{plugin_name}' 已经加载，无需重复注册。")
        return True

    # 如果插件尚未注册，则开始搜索流程
    if plugin_name not in plugin_manager.list_registered_plugins():
        logger.info(f"插件 '{plugin_name}' 未注册，开始在插件目录中搜索...")
        found_path = None

        # 遍历所有配置的插件目录
        for directory in plugin_manager.plugin_directories:
            potential_path = os.path.join(directory, plugin_name)
            # 检查是否存在与插件同名的目录
            if os.path.isdir(potential_path):
                found_path = potential_path
                break

        # 如果未找到插件目录，则报告错误
        if not found_path:
            logger.error(f"在所有插件目录中都未找到名为 '{plugin_name}' 的插件。")
            return False

        # 检查插件的核心 'plugin.py' 文件是否存在
        plugin_file = os.path.join(found_path, "plugin.py")
        if not os.path.exists(plugin_file):
            logger.error(f"在插件目录 '{found_path}' 中未找到核心的 plugin.py 文件。")
            return False

        # 尝试从文件加载插件模块
        module = plugin_manager._load_plugin_module_file(plugin_file)
        if not module:
            logger.error(f"从文件 '{plugin_file}' 加载插件模块失败。")
            return False

        # 验证模块加载后，插件是否已成功注册
        if plugin_name not in plugin_manager.list_registered_plugins():
            logger.error(f"插件 '{plugin_name}' 在加载模块后依然未能成功注册。请检查插件定义。")
            return False

        logger.info(f"插件 '{plugin_name}' 已成功发现并注册。")

    # 根据参数决定是否在注册后立即加载插件
    if load_after_register:
        logger.info(f"正在加载插件 '{plugin_name}'...")
        status, _ = plugin_manager.load_registered_plugin_classes(plugin_name)
        return status

    return True


# --------------------------------------------------------------------------------
# Section 2: 插件状态管理 (Plugin State Management)
# --------------------------------------------------------------------------------
# 该部分包含控制插件整体启用/禁用状态的功能。


async def load_plugin(plugin_name: str) -> bool:
    """
    加载一个已注册但未加载的插件。

    Args:
        plugin_name (str): 要加载的插件名称。

    Returns:
        bool: 如果插件成功加载，则为 True。

    Raises:
        ValueError: 如果插件未注册或已经加载。
    """
    # 检查插件是否已经加载
    if plugin_name in plugin_manager.list_loaded_plugins():
        logger.warning(f"插件 '{plugin_name}' 已经加载。")
        return True

    # 检查插件是否已注册
    if plugin_name not in plugin_manager.list_registered_plugins():
        raise ValueError(f"插件 '{plugin_name}' 未注册，无法加载。")

    # 尝试加载插件
    success, _ = plugin_manager.load_registered_plugin_classes(plugin_name)
    if success:
        logger.info(f"插件 '{plugin_name}' 加载成功。")
    else:
        logger.error(f"插件 '{plugin_name}' 加载失败。")
    
    return success


async def enable_plugin(plugin_name: str) -> bool:
    """
    启用一个已禁用的插件。

    如果插件已经启用，直接返回 True。
    启用插件会同时尝试加载其所有组件。

    Args:
        plugin_name (str): 要启用的插件名称。

    Returns:
        bool: 如果插件成功启用，则为 True。
    """
    plugin_instance = plugin_manager.get_plugin_instance(plugin_name)

    if not plugin_instance:
        # 插件未加载，尝试从注册表加载
        if plugin_name not in plugin_manager.list_registered_plugins():
            logger.error(f"插件 '{plugin_name}' 未注册，无法启用。")
            return False

        # 尝试加载插件
        success, _ = plugin_manager.load_registered_plugin_classes(plugin_name)
        if not success:
            logger.error(f"加载插件 '{plugin_name}' 失败。")
            return False

        logger.info(f"插件 '{plugin_name}' 已成功启用并加载。")
        return True

    # 如果插件已经启用
    if plugin_instance.enable_plugin:
        logger.info(f"插件 '{plugin_name}' 已经处于启用状态。")
        return True

    # 设置插件为启用状态
    plugin_instance.enable_plugin = True
    logger.info(f"插件 '{plugin_name}' 已启用。")
    return True


async def disable_plugin(plugin_name: str,) -> bool:
    """
    禁用一个插件。

    禁用插件不会卸载它，只会标记为禁用状态。
    包含对 Chatter 组件的保护机制，防止禁用最后一个启用的 Chatter。

    Args:
        plugin_name (str): 要禁用的插件名称。

    Returns:
        bool: 如果插件成功禁用，则为 True。
    """
    plugin_instance = plugin_manager.get_plugin_instance(plugin_name)

    if not plugin_instance:
        logger.warning(f"插件 '{plugin_name}' 未加载，无需禁用。")
        return True

    # Chatter 保护检查：确保系统中至少有一个 Chatter 组件处于启用状态
    try:
        from src.plugin_system.base.component_types import ComponentType
        from src.plugin_system.core.component_registry import component_registry

        # 获取该插件的所有组件
        plugin_info = component_registry.get_plugin_info(plugin_name)
        if plugin_info:
            # 检查插件是否包含 Chatter 组件
            has_chatter = any(
                comp.component_type == ComponentType.CHATTER
                for comp in plugin_info.components
            )

            if has_chatter:
                # 获取所有启用的 Chatter 组件
                enabled_chatters = component_registry.get_enabled_components_by_type(ComponentType.CHATTER)

                # 统计该插件中启用的 Chatter 数量
                plugin_enabled_chatters = [
                    comp.name for comp in plugin_info.components
                    if comp.component_type == ComponentType.CHATTER
                    and comp.name in enabled_chatters
                ]

                # 如果禁用此插件会导致没有可用的 Chatter，则阻止操作
                if len(enabled_chatters) <= len(plugin_enabled_chatters):
                    logger.warning(
                        f"操作被阻止：禁用插件 '{plugin_name}' 将导致系统中没有可用的 Chatter 组件。"
                        f"至少需要保持一个 Chatter 组件处于启用状态。"
                    )
                    return False
    except Exception as e:
        logger.warning(f"检查 Chatter 保护机制时发生错误: {e}")
        # 即使检查失败，也继续执行禁用操作（降级处理）

    # 设置插件为禁用状态
    plugin_instance.enable_plugin = False
    logger.info(f"插件 '{plugin_name}' 已禁用。")
    return True


async def unload_plugin(plugin_name: str) -> bool:
    """
    完全卸载一个插件。

    这会从内存中移除插件及其所有组件。与禁用不同，卸载后需要重新加载才能使用。

    Args:
        plugin_name (str): 要卸载的插件名称。

    Returns:
        bool: 如果插件成功卸载，则为 True。
    """
    if plugin_name not in plugin_manager.list_loaded_plugins():
        logger.warning(f"插件 '{plugin_name}' 未加载，无需卸载。")
        return True

    return await plugin_manager.remove_registered_plugin(plugin_name)


def is_plugin_enabled(plugin_name: str) -> bool:
    """
    检查插件是否处于启用状态。

    Args:
        plugin_name (str): 要检查的插件名称。

    Returns:
        bool: 如果插件已启用，则为 True；如果未加载或已禁用，则为 False。
    """
    plugin_instance = plugin_manager.get_plugin_instance(plugin_name)
    if not plugin_instance:
        return False
    return plugin_instance.enable_plugin


def is_plugin_loaded(plugin_name: str) -> bool:
    """
    快速检查一个插件当前是否已成功加载。

    Args:
        plugin_name (str): 要检查的插件名称。

    Returns:
        bool: 如果插件已加载，则为 True，否则为 False。
    """
    return plugin_name in plugin_manager.list_loaded_plugins()


def list_loaded_plugins() -> list[str]:
    """
    列出所有已加载的插件名称。

    Returns:
        list[str]: 已加载插件的名称列表。
    """
    return plugin_manager.list_loaded_plugins()


def list_registered_plugins() -> list[str]:
    """
    列出所有已注册的插件名称。

    Returns:
        list[str]: 已注册插件的名称列表。
    """
    return plugin_manager.list_registered_plugins()


def list_failed_plugins() -> dict[str, str]:
    """
    获取所有加载失败的插件及其错误信息。

    Returns:
        dict[str, str]: 插件名称到错误信息的映射。
    """
    return plugin_manager.failed_plugins.copy()


def get_plugin_instance(plugin_name: str):
    """
    获取插件实例。

    Args:
        plugin_name (str): 插件名称。

    Returns:
        BasePlugin | None: 插件实例，如果不存在则返回 None。
    """
    return plugin_manager.get_plugin_instance(plugin_name)

