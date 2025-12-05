# mem_monitor.py
"""
内存监控工具模块

用于监控和诊断 MoFox-Bot 的内存使用情况，包括：
- RSS/VMS 内存使用追踪
- tracemalloc 内存分配差异分析
- 对象类型增长监控 (objgraph)
- 类型内存占用分析 (Pympler)

通过环境变量 MEM_MONITOR_ENABLED 控制是否启用（默认禁用）
日志输出到独立文件 logs/mem_monitor.log
"""

import logging
import os
import threading
import time
import tracemalloc
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

# objgraph 是可选依赖，用于对象增长监控
try:
    import objgraph
    OBJGRAPH_AVAILABLE = True
except ImportError:
    OBJGRAPH_AVAILABLE = False
    objgraph = None  # type: ignore[assignment]

# pympler 是可选依赖，用于类型内存分析
try:
    from pympler import muppy, summary
    PYMPLER_AVAILABLE = True
except ImportError:
    PYMPLER_AVAILABLE = False
    muppy = None
    summary = None

if TYPE_CHECKING:
    from psutil import Process

# 创建独立的内存监控日志器
def _setup_mem_logger() -> logging.Logger:
    """设置独立的内存监控日志器，输出到单独的文件"""
    logger = logging.getLogger("mem_monitor")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 不传播到父日志器，避免污染主日志
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 文件处理器 - 带日期的日志文件，支持轮转
    log_file = log_dir / f"mem_monitor_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    
    # 格式化器
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    
    # 控制台处理器 - 只输出重要信息
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


logger = _setup_mem_logger()

# 启动时记录可选依赖的可用性
if not OBJGRAPH_AVAILABLE:
    logger.warning("objgraph 未安装，对象增长分析功能不可用 (pip install objgraph)")
if not PYMPLER_AVAILABLE:
    logger.warning("pympler 未安装，类型内存分析功能不可用 (pip install Pympler)")

_process: "Process" = psutil.Process()
_last_snapshot: tracemalloc.Snapshot | None = None
_last_type_summary: list | None = None
_monitor_thread: threading.Thread | None = None
_stop_event: threading.Event = threading.Event()

# 环境变量控制是否启用，防止所有环境一起开
MEM_MONITOR_ENABLED = False


def start_tracemalloc(max_frames: int = 25) -> None:
    """启动 tracemalloc 内存追踪

    Args:
        max_frames: 追踪的最大栈帧数，越大越详细但开销越大
    """
    if not tracemalloc.is_tracing():
        tracemalloc.start(max_frames)
        logger.info("tracemalloc started with max_frames=%s", max_frames)
    else:
        logger.info("tracemalloc already started")


def stop_tracemalloc() -> None:
    """停止 tracemalloc 内存追踪"""
    if tracemalloc.is_tracing():
        tracemalloc.stop()
        logger.info("tracemalloc stopped")


def log_rss(tag: str = "periodic") -> dict[str, float]:
    """记录当前进程的 RSS 和 VMS 内存使用

    Args:
        tag: 日志标签，用于区分不同的采样点

    Returns:
        包含 rss_mb 和 vms_mb 的字典
    """
    mem = _process.memory_info()
    rss_mb = mem.rss / (1024 * 1024)
    vms_mb = mem.vms / (1024 * 1024)
    logger.info("[MEM %s] RSS=%.1f MiB, VMS=%.1f MiB", tag, rss_mb, vms_mb)
    return {"rss_mb": rss_mb, "vms_mb": vms_mb}


def log_tracemalloc_diff(tag: str = "periodic", limit: int = 20):
    global _last_snapshot

    if not tracemalloc.is_tracing():
        logger.warning("tracemalloc is not tracing, skip diff")
        return

    snapshot = tracemalloc.take_snapshot()
    if _last_snapshot is None:
        logger.info("[TM %s] first snapshot captured", tag)
        _last_snapshot = snapshot
        return

    logger.info("[TM %s] top %s memory diffs (by traceback):", tag, limit)
    top_stats = snapshot.compare_to(_last_snapshot, "traceback")

    for idx, stat in enumerate(top_stats[:limit], start=1):
        logger.info(
            "[TM %s] #%d: size_diff=%s, count_diff=%s",
            tag, idx, stat.size_diff, stat.count_diff
        )
        # 打完整调用栈
        for line in stat.traceback.format():
            logger.info("[TM %s]    %s", tag, line)

    _last_snapshot = snapshot


def log_object_growth(limit: int = 20) -> None:
    """使用 objgraph 查看最近一段时间哪些对象类型数量增长

    Args:
        limit: 显示的最大增长类型数
    """
    if not OBJGRAPH_AVAILABLE or objgraph is None:
        logger.warning("objgraph not available, skipping object growth analysis")
        return
    
    logger.info("==== Objgraph growth (top %s) ====", limit)
    try:
        # objgraph.show_growth 默认输出到 stdout，需要捕获输出
        import io
        import sys
        
        # 捕获 stdout
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            objgraph.show_growth(limit=limit)
        finally:
            sys.stdout = old_stdout
        
        output = buffer.getvalue()
        if output.strip():
            for line in output.strip().split("\n"):
                logger.info("[OG] %s", line)
        else:
            logger.info("[OG] No object growth detected")
    except Exception:
        logger.exception("objgraph.show_growth failed")


def log_type_memory_diff() -> None:
    """使用 Pympler 查看各类型对象占用的内存变化"""
    global _last_type_summary
    
    if not PYMPLER_AVAILABLE or muppy is None or summary is None:
        logger.warning("pympler not available, skipping type memory analysis")
        return
    
    import io
    import sys
    
    all_objects = muppy.get_objects()
    curr = summary.summarize(all_objects)

    # 捕获 Pympler 的输出（summary.print_ 也是输出到 stdout）
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    try:
        if _last_type_summary is None:
            logger.info("==== Pympler initial type summary ====")
            summary.print_(curr)
        else:
            logger.info("==== Pympler type memory diff ====")
            diff = summary.get_diff(_last_type_summary, curr)
            summary.print_(diff)
    finally:
        sys.stdout = old_stdout

    output = buffer.getvalue()
    if output.strip():
        for line in output.strip().split("\n"):
            logger.info("[PY] %s", line)

    _last_type_summary = curr


def periodic_mem_monitor(interval_sec: int = 900, tracemalloc_limit: int = 20, objgraph_limit: int = 20) -> None:
    """后台循环：定期记录 RSS、tracemalloc diff、对象增长情况

    Args:
        interval_sec: 采样间隔（秒）
        tracemalloc_limit: tracemalloc 差异显示限制
        objgraph_limit: objgraph 增长显示限制
    """
    if not MEM_MONITOR_ENABLED:
        logger.info("Memory monitor disabled via MEM_MONITOR_ENABLED=0")
        return

    start_tracemalloc()

    logger.info("Memory monitor thread started, interval=%s sec", interval_sec)

    counter = 0
    while not _stop_event.is_set():
        # 使用 Event.wait 替代 time.sleep，支持优雅退出
        if _stop_event.wait(timeout=interval_sec):
            break

        try:
            counter += 1
            log_rss("periodic")
            log_tracemalloc_diff("periodic", limit=tracemalloc_limit)
            log_object_growth(limit=objgraph_limit)
            if counter % 3 == 0:
                log_type_memory_diff()
        except Exception:
            logger.exception("Memory monitor iteration failed")

    logger.info("Memory monitor thread stopped")


def start_background_monitor(interval_sec: int = 300, tracemalloc_limit: int = 20, objgraph_limit: int = 20) -> bool:
    """在项目入口调用，用线程避免阻塞主 event loop

    Args:
        interval_sec: 采样间隔（秒）
        tracemalloc_limit: tracemalloc 差异显示限制
        objgraph_limit: objgraph 增长显示限制

    Returns:
        是否成功启动监控线程
    """
    global _monitor_thread

    if not MEM_MONITOR_ENABLED:
        logger.info("Memory monitor not started (disabled via MEM_MONITOR_ENABLED env var).")
        return False

    if _monitor_thread is not None and _monitor_thread.is_alive():
        logger.warning("Memory monitor thread already running")
        return True

    _stop_event.clear()
    _monitor_thread = threading.Thread(
        target=periodic_mem_monitor,
        kwargs={
            "interval_sec": interval_sec,
            "tracemalloc_limit": tracemalloc_limit,
            "objgraph_limit": objgraph_limit,
        },
        daemon=True,
        name="MemoryMonitorThread",
    )
    _monitor_thread.start()
    logger.info("Memory monitor thread created (interval=%s sec)", interval_sec)
    return True


def stop_background_monitor(timeout: float = 5.0) -> None:
    """停止后台内存监控线程

    Args:
        timeout: 等待线程退出的超时时间（秒）
    """
    global _monitor_thread

    if _monitor_thread is None or not _monitor_thread.is_alive():
        logger.debug("Memory monitor thread not running")
        return

    logger.info("Stopping memory monitor thread...")
    _stop_event.set()
    _monitor_thread.join(timeout=timeout)

    if _monitor_thread.is_alive():
        logger.warning("Memory monitor thread did not stop within timeout")
    else:
        logger.info("Memory monitor thread stopped successfully")

    _monitor_thread = None


def manual_dump(tag: str = "manual") -> dict:
    """手动触发一次采样，可以挂在 HTTP /debug/mem 上

    Args:
        tag: 日志标签

    Returns:
        包含内存信息的字典
    """
    logger.info("Manual memory dump started: %s", tag)
    mem_info = log_rss(tag)
    log_tracemalloc_diff(tag)
    log_object_growth()
    log_type_memory_diff()
    logger.info("Manual memory dump finished: %s", tag)
    return mem_info


def debug_leak_for_type(type_name: str, max_depth: int = 5, filename: str | None = None) -> bool:
    """对某个可疑类型画引用图，看是谁抓着它不放

    建议只在本地/测试环境用，这个可能比较慢。

    Args:
        type_name: 要调试的类型名（如 'MySession'）
        max_depth: 引用图的最大深度
        filename: 输出文件名，默认为 "{type_name}_backrefs.png"

    Returns:
        是否成功生成引用图
    """
    if not OBJGRAPH_AVAILABLE or objgraph is None:
        logger.warning("objgraph not available, cannot generate backrefs graph")
        return False
    
    if filename is None:
        filename = f"{type_name}_backrefs.png"

    try:
        objs = objgraph.by_type(type_name)
        if not objs:
            logger.info("No objects of type %s", type_name)
            return False

        # 随便拿几个代表对象看引用链
        roots = objs[:3]
        logger.info(
            "Generating backrefs graph for %s (num_roots=%s, max_depth=%s, file=%s)",
            type_name,
            len(roots),
            max_depth,
            filename,
        )
        objgraph.show_backrefs(
            roots,
            max_depth=max_depth,
            filename=filename,
        )
        logger.info("Backrefs graph generated: %s", filename)
        return True
    except Exception:
        logger.exception("debug_leak_for_type(%s) failed", type_name)
        return False


def get_memory_stats() -> dict:
    """获取当前内存统计信息

    Returns:
        包含各项内存指标的字典
    """
    mem = _process.memory_info()
    return {
        "rss_mb": mem.rss / (1024 * 1024),
        "vms_mb": mem.vms / (1024 * 1024),
        "tracemalloc_enabled": tracemalloc.is_tracing(),
        "monitor_thread_alive": _monitor_thread is not None and _monitor_thread.is_alive(),
    }
