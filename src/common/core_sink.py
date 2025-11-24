"""
从 src.main 导出 core_sink 的辅助函数

由于 src.main 中实际使用的是 InProcessCoreSink，
我们需要创建一个全局访问点
"""

from mofox_bus import CoreSink, InProcessCoreSink

_global_core_sink: CoreSink | None = None


def set_core_sink(sink: CoreSink) -> None:
    """设置全局 core sink"""
    global _global_core_sink
    _global_core_sink = sink


def get_core_sink() -> CoreSink:
    """获取全局 core sink"""
    global _global_core_sink
    if _global_core_sink is None:
        raise RuntimeError("Core sink 尚未初始化")
    return _global_core_sink


async def push_outgoing(envelope) -> None:
    """将消息推送到 core sink 的 outgoing 通道"""
    sink = get_core_sink()
    push = getattr(sink, "push_outgoing", None)
    if push is None:
        raise RuntimeError("当前 core sink 不支持 push_outgoing 方法")
    await push(envelope)

__all__ = ["set_core_sink", "get_core_sink", "push_outgoing"]
