"""
插件热重载模块

使用 Watchdog 监听插件目录变化，自动重载插件
"""

import os
import sys
import time
import importlib
from pathlib import Path
from threading import Thread
from typing import Dict, Set, List, Optional, Tuple

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.common.logger import get_logger
from .plugin_manager import plugin_manager

logger = get_logger("plugin_hot_reload")


class PluginFileHandler(FileSystemEventHandler):
    """插件文件变化处理器"""

    def __init__(self, hot_reload_manager):
        super().__init__()
        self.hot_reload_manager = hot_reload_manager
        self.pending_reloads: Set[str] = set()  # 待重载的插件名称
        self.last_reload_time: Dict[str, float] = {}  # 上次重载时间
        self.debounce_delay = 2.0  # 增加防抖延迟到2秒，确保文件写入完成
        self.file_change_cache: Dict[str, float] = {}  # 文件变化缓存

    def on_modified(self, event):
        """文件修改事件"""
        if not event.is_directory:
            file_path = str(event.src_path)
            if file_path.endswith((".py", ".toml")):
                self._handle_file_change(file_path, "modified")

    def on_created(self, event):
        """文件创建事件"""
        if not event.is_directory:
            file_path = str(event.src_path)
            if file_path.endswith((".py", ".toml")):
                self._handle_file_change(file_path, "created")

    def on_deleted(self, event):
        """文件删除事件"""
        if not event.is_directory:
            file_path = str(event.src_path)
            if file_path.endswith((".py", ".toml")):
                self._handle_file_change(file_path, "deleted")

    def _handle_file_change(self, file_path: str, change_type: str):
        """处理文件变化"""
        try:
            # 获取插件名称
            plugin_info = self._get_plugin_info_from_path(file_path)
            if not plugin_info:
                return

            plugin_name, source_type = plugin_info
            current_time = time.time()

            # 文件变化缓存，避免重复处理同一文件的快速连续变化
            file_cache_key = f"{file_path}_{change_type}"
            last_file_time = self.file_change_cache.get(file_cache_key, 0)
            if current_time - last_file_time < 0.5:  # 0.5秒内的重复文件变化忽略
                return
            self.file_change_cache[file_cache_key] = current_time

            # 插件级别的防抖处理
            last_plugin_time = self.last_reload_time.get(plugin_name, 0)
            if current_time - last_plugin_time < self.debounce_delay:
                # 如果在防抖期内，更新待重载标记但不立即处理
                self.pending_reloads.add(plugin_name)
                return

            file_name = Path(file_path).name
            logger.info(f"📁 检测到插件文件变化: {file_name} ({change_type}) [{source_type}] -> {plugin_name}")

            # 如果是删除事件，处理关键文件删除
            if change_type == "deleted":
                # 解析实际的插件名称
                actual_plugin_name = self.hot_reload_manager._resolve_plugin_name(plugin_name)

                if file_name == "plugin.py":
                    if actual_plugin_name in plugin_manager.loaded_plugins:
                        logger.info(
                            f"🗑️ 插件主文件被删除，卸载插件: {plugin_name} -> {actual_plugin_name} [{source_type}]"
                        )
                        self.hot_reload_manager._unload_plugin(actual_plugin_name)
                    else:
                        logger.info(
                            f"🗑️ 插件主文件被删除，但插件未加载: {plugin_name} -> {actual_plugin_name} [{source_type}]"
                        )
                    return
                elif file_name in ("manifest.toml", "_manifest.json"):
                    if actual_plugin_name in plugin_manager.loaded_plugins:
                        logger.info(
                            f"🗑️ 插件配置文件被删除，卸载插件: {plugin_name} -> {actual_plugin_name} [{source_type}]"
                        )
                        self.hot_reload_manager._unload_plugin(actual_plugin_name)
                    else:
                        logger.info(
                            f"🗑️ 插件配置文件被删除，但插件未加载: {plugin_name} -> {actual_plugin_name} [{source_type}]"
                        )
                    return

            # 对于修改和创建事件，都进行重载
            # 添加到待重载列表
            self.pending_reloads.add(plugin_name)
            self.last_reload_time[plugin_name] = current_time

            # 延迟重载，确保文件写入完成
            reload_thread = Thread(
                target=self._delayed_reload, args=(plugin_name, source_type, current_time), daemon=True
            )
            reload_thread.start()

        except Exception as e:
            logger.error(f"❌ 处理文件变化时发生错误: {e}", exc_info=True)

    def _delayed_reload(self, plugin_name: str, source_type: str, trigger_time: float):
        """延迟重载插件"""
        try:
            # 等待文件写入完成
            time.sleep(self.debounce_delay)

            # 检查是否还需要重载（可能在等待期间有更新的变化）
            if plugin_name not in self.pending_reloads:
                return

            # 检查是否有更新的重载请求
            if self.last_reload_time.get(plugin_name, 0) > trigger_time:
                return

            self.pending_reloads.discard(plugin_name)
            logger.info(f"🔄 开始延迟重载插件: {plugin_name} [{source_type}]")

            # 执行深度重载
            success = self.hot_reload_manager._deep_reload_plugin(plugin_name)
            if success:
                logger.info(f"✅ 插件重载成功: {plugin_name} [{source_type}]")
            else:
                logger.error(f"❌ 插件重载失败: {plugin_name} [{source_type}]")

        except Exception as e:
            logger.error(f"❌ 延迟重载插件 {plugin_name} 时发生错误: {e}", exc_info=True)

    def _get_plugin_info_from_path(self, file_path: str) -> Optional[Tuple[str, str]]:
        """从文件路径获取插件信息

        Returns:
            tuple[插件名称, 源类型] 或 None
        """
        try:
            path = Path(file_path)

            # 检查是否在任何一个监听的插件目录中
            for watch_dir in self.hot_reload_manager.watch_directories:
                plugin_root = Path(watch_dir)
                if path.is_relative_to(plugin_root):
                    # 确定源类型
                    if "src" in str(plugin_root):
                        source_type = "built-in"
                    else:
                        source_type = "external"

                    # 获取插件目录名（插件名）
                    relative_path = path.relative_to(plugin_root)
                    if len(relative_path.parts) == 0:
                        continue

                    plugin_name = relative_path.parts[0]

                    # 确认这是一个有效的插件目录
                    plugin_dir = plugin_root / plugin_name
                    if plugin_dir.is_dir():
                        # 检查是否有插件主文件或配置文件
                        has_plugin_py = (plugin_dir / "plugin.py").exists()
                        has_manifest = (plugin_dir / "manifest.toml").exists() or (
                            plugin_dir / "_manifest.json"
                        ).exists()

                        if has_plugin_py or has_manifest:
                            return plugin_name, source_type

            return None

        except Exception:
            return None


class PluginHotReloadManager:
    """插件热重载管理器"""

    def __init__(self, watch_directories: Optional[List[str]] = None):
        if watch_directories is None:
            # 默认监听两个目录：根目录下的 plugins 和 src 下的插件目录
            self.watch_directories = [
                os.path.join(os.getcwd(), "plugins"),  # 外部插件目录
                os.path.join(os.getcwd(), "src", "plugins", "built_in"),  # 内置插件目录
            ]
        else:
            self.watch_directories = watch_directories

        self.observers = []
        self.file_handlers = []
        self.is_running = False

        # 确保监听目录存在
        for watch_dir in self.watch_directories:
            if not os.path.exists(watch_dir):
                os.makedirs(watch_dir, exist_ok=True)
                logger.info(f"📁 创建插件监听目录: {watch_dir}")

    def start(self):
        """启动热重载监听"""
        if self.is_running:
            logger.warning("插件热重载已经在运行中")
            return

        try:
            # 为每个监听目录创建独立的观察者
            for watch_dir in self.watch_directories:
                observer = Observer()
                file_handler = PluginFileHandler(self)

                observer.schedule(file_handler, watch_dir, recursive=True)

                observer.start()
                self.observers.append(observer)
                self.file_handlers.append(file_handler)

            self.is_running = True

            # 打印监听的目录信息
            dir_info = []
            for watch_dir in self.watch_directories:
                if "src" in watch_dir:
                    dir_info.append(f"{watch_dir} (内置插件)")
                else:
                    dir_info.append(f"{watch_dir} (外部插件)")

            logger.info("🚀 插件热重载已启动，监听目录:")
            for info in dir_info:
                logger.info(f"  📂 {info}")

        except Exception as e:
            logger.error(f"❌ 启动插件热重载失败: {e}")
            self.stop()  # 清理已创建的观察者
            self.is_running = False

    def stop(self):
        """停止热重载监听"""
        if not self.is_running and not self.observers:
            return

        # 停止所有观察者
        for observer in self.observers:
            try:
                observer.stop()
                observer.join()
            except Exception as e:
                logger.error(f"❌ 停止观察者时发生错误: {e}")

        self.observers.clear()
        self.file_handlers.clear()
        self.is_running = False
        logger.info("🛑 插件热重载已停止")

    def _reload_plugin(self, plugin_name: str):
        """重载指定插件（简单重载）"""
        try:
            # 解析实际的插件名称
            actual_plugin_name = self._resolve_plugin_name(plugin_name)
            logger.info(f"🔄 开始简单重载插件: {plugin_name} -> {actual_plugin_name}")

            if plugin_manager.reload_plugin(actual_plugin_name):
                logger.info(f"✅ 插件简单重载成功: {actual_plugin_name}")
                return True
            else:
                logger.error(f"❌ 插件简单重载失败: {actual_plugin_name}")
                return False

        except Exception as e:
            logger.error(f"❌ 重载插件 {plugin_name} 时发生错误: {e}", exc_info=True)
            return False

    @staticmethod
    def _resolve_plugin_name(folder_name: str) -> str:
        """
        将文件夹名称解析为实际的插件名称
        通过检查插件管理器中的路径映射来找到对应的插件名
        """
        # 首先检查是否直接匹配
        if folder_name in plugin_manager.plugin_classes:
            logger.debug(f"🔍 直接匹配插件名: {folder_name}")
            return folder_name

        # 如果没有直接匹配，搜索路径映射，并优先返回在插件类中存在的名称
        matched_plugins = []
        for plugin_name, plugin_path in plugin_manager.plugin_paths.items():
            # 检查路径是否包含该文件夹名
            if folder_name in plugin_path:
                matched_plugins.append((plugin_name, plugin_path))

        # 在匹配的插件中，优先选择在插件类中存在的
        for plugin_name, plugin_path in matched_plugins:
            if plugin_name in plugin_manager.plugin_classes:
                logger.debug(f"🔍 文件夹名 '{folder_name}' 映射到插件名 '{plugin_name}' (路径: {plugin_path})")
                return plugin_name

        # 如果还是没找到在插件类中存在的，返回第一个匹配项
        if matched_plugins:
            plugin_name, plugin_path = matched_plugins[0]
            logger.warning(f"⚠️ 文件夹 '{folder_name}' 映射到 '{plugin_name}'，但该插件类不存在")
            return plugin_name

        # 如果还是没找到，返回原文件夹名
        logger.warning(f"⚠️ 无法找到文件夹 '{folder_name}' 对应的插件名，使用原名称")
        return folder_name

    def _deep_reload_plugin(self, plugin_name: str):
        """深度重载指定插件（清理模块缓存）"""
        try:
            # 解析实际的插件名称
            actual_plugin_name = self._resolve_plugin_name(plugin_name)
            logger.info(f"🔄 开始深度重载插件: {plugin_name} -> {actual_plugin_name}")

            # 强制清理相关模块缓存
            self._force_clear_plugin_modules(plugin_name)

            # 使用插件管理器的强制重载功能
            success = plugin_manager.force_reload_plugin(actual_plugin_name)

            if success:
                logger.info(f"✅ 插件深度重载成功: {actual_plugin_name}")
                return True
            else:
                logger.error(f"❌ 插件深度重载失败，尝试简单重载: {actual_plugin_name}")
                # 如果深度重载失败，尝试简单重载
                return self._reload_plugin(actual_plugin_name)

        except Exception as e:
            logger.error(f"❌ 深度重载插件 {plugin_name} 时发生错误: {e}", exc_info=True)
            # 出错时尝试简单重载
            return self._reload_plugin(plugin_name)

    @staticmethod
    def _force_clear_plugin_modules(plugin_name: str):
        """强制清理插件相关的模块缓存"""

        # 找到所有相关的模块名
        modules_to_remove = []
        plugin_module_prefix = f"src.plugins.built_in.{plugin_name}"

        for module_name in list(sys.modules.keys()):
            if plugin_module_prefix in module_name:
                modules_to_remove.append(module_name)

        # 删除模块缓存
        for module_name in modules_to_remove:
            if module_name in sys.modules:
                logger.debug(f"🗑️ 清理模块缓存: {module_name}")
                del sys.modules[module_name]

    @staticmethod
    def _force_reimport_plugin(plugin_name: str):
        """强制重新导入插件（委托给插件管理器）"""
        try:
            # 使用插件管理器的重载功能
            success = plugin_manager.reload_plugin(plugin_name)
            return success

        except Exception as e:
            logger.error(f"❌ 强制重新导入插件 {plugin_name} 时发生错误: {e}", exc_info=True)
            return False

    @staticmethod
    def _unload_plugin(plugin_name: str):
        """卸载指定插件"""
        try:
            logger.info(f"🗑️ 开始卸载插件: {plugin_name}")

            if plugin_manager.unload_plugin(plugin_name):
                logger.info(f"✅ 插件卸载成功: {plugin_name}")
                return True
            else:
                logger.error(f"❌ 插件卸载失败: {plugin_name}")
                return False

        except Exception as e:
            logger.error(f"❌ 卸载插件 {plugin_name} 时发生错误: {e}", exc_info=True)
            return False

    def reload_all_plugins(self):
        """重载所有插件"""
        try:
            logger.info("🔄 开始深度重载所有插件...")

            # 获取当前已加载的插件列表
            loaded_plugins = list(plugin_manager.loaded_plugins.keys())

            success_count = 0
            fail_count = 0

            for plugin_name in loaded_plugins:
                logger.info(f"🔄 重载插件: {plugin_name}")
                if self._deep_reload_plugin(plugin_name):
                    success_count += 1
                else:
                    fail_count += 1

            logger.info(f"✅ 插件重载完成: 成功 {success_count} 个，失败 {fail_count} 个")

            # 清理全局缓存
            importlib.invalidate_caches()

        except Exception as e:
            logger.error(f"❌ 重载所有插件时发生错误: {e}", exc_info=True)

    def force_reload_plugin(self, plugin_name: str):
        """手动强制重载指定插件（委托给插件管理器）"""
        try:
            logger.info(f"🔄 手动强制重载插件: {plugin_name}")

            # 清理待重载列表中的该插件（避免重复重载）
            for handler in self.file_handlers:
                handler.pending_reloads.discard(plugin_name)

            # 使用插件管理器的强制重载功能
            success = plugin_manager.force_reload_plugin(plugin_name)

            if success:
                logger.info(f"✅ 手动强制重载成功: {plugin_name}")
            else:
                logger.error(f"❌ 手动强制重载失败: {plugin_name}")

            return success

        except Exception as e:
            logger.error(f"❌ 手动强制重载插件 {plugin_name} 时发生错误: {e}", exc_info=True)
            return False

    def add_watch_directory(self, directory: str):
        """添加新的监听目录"""
        if directory in self.watch_directories:
            logger.info(f"目录 {directory} 已在监听列表中")
            return

        # 确保目录存在
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 创建插件监听目录: {directory}")

        self.watch_directories.append(directory)

        # 如果热重载正在运行，为新目录创建观察者
        if self.is_running:
            try:
                observer = Observer()
                file_handler = PluginFileHandler(self)

                observer.schedule(file_handler, directory, recursive=True)

                observer.start()
                self.observers.append(observer)
                self.file_handlers.append(file_handler)

                logger.info(f"📂 已添加新的监听目录: {directory}")

            except Exception as e:
                logger.error(f"❌ 添加监听目录 {directory} 失败: {e}")
                self.watch_directories.remove(directory)

    def get_status(self) -> dict:
        """获取热重载状态"""
        pending_reloads = set()
        if self.file_handlers:
            for handler in self.file_handlers:
                pending_reloads.update(handler.pending_reloads)

        return {
            "is_running": self.is_running,
            "watch_directories": self.watch_directories,
            "active_observers": len(self.observers),
            "loaded_plugins": len(plugin_manager.loaded_plugins),
            "failed_plugins": len(plugin_manager.failed_plugins),
            "pending_reloads": list(pending_reloads),
            "debounce_delay": self.file_handlers[0].debounce_delay if self.file_handlers else 0,
        }

    @staticmethod
    def clear_all_caches():
        """清理所有Python模块缓存"""
        try:
            logger.info("🧹 开始清理所有Python模块缓存...")

            # 重新扫描所有插件目录，这会重新加载模块
            plugin_manager.rescan_plugin_directory()
            logger.info("✅ 模块缓存清理完成")

        except Exception as e:
            logger.error(f"❌ 清理模块缓存时发生错误: {e}", exc_info=True)


# 全局热重载管理器实例
hot_reload_manager = PluginHotReloadManager()
