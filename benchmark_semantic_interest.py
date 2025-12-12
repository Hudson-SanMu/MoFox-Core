"""语义兴趣度评分器性能测试

对比测试：
1. 原始 sklearn 路径 vs FastScorer
2. 单条评分 vs 批处理
3. 同步 vs 异步
"""

import asyncio
import time
from pathlib import Path

# 测试样本
SAMPLE_TEXTS = [
    "今天天气真好",
    "这个游戏太好玩了！",
    "无聊死了",
    "我对这个话题很感兴趣",
    "能不能聊点别的",
    "哇这个真的很厉害",
    "你好",
    "有人在吗",
    "这个问题很有深度",
    "随便说说",
    "真是太棒了，我非常喜欢",
    "算了算了不想说了",
    "来聊聊最近的新闻吧",
    "emmmm",
    "哈哈哈哈",
    "666",
]


def benchmark_sklearn_scorer(model_path: str, iterations: int = 100):
    """测试原始 sklearn 评分器"""
    from src.chat.semantic_interest.runtime_scorer import SemanticInterestScorer
    
    scorer = SemanticInterestScorer(model_path, use_fast_scorer=False)
    scorer.load()
    
    # 预热
    for text in SAMPLE_TEXTS[:3]:
        scorer.score(text)
    
    # 单条评分测试
    start = time.perf_counter()
    for _ in range(iterations):
        for text in SAMPLE_TEXTS:
            scorer.score(text)
    single_time = time.perf_counter() - start
    total_single = iterations * len(SAMPLE_TEXTS)
    
    # 批量评分测试
    start = time.perf_counter()
    for _ in range(iterations):
        scorer.score_batch(SAMPLE_TEXTS)
    batch_time = time.perf_counter() - start
    total_batch = iterations * len(SAMPLE_TEXTS)
    
    return {
        "mode": "sklearn",
        "single_total_time": single_time,
        "single_avg_ms": single_time / total_single * 1000,
        "single_qps": total_single / single_time,
        "batch_total_time": batch_time,
        "batch_avg_ms": batch_time / total_batch * 1000,
        "batch_qps": total_batch / batch_time,
    }


def benchmark_fast_scorer(model_path: str, iterations: int = 100):
    """测试 FastScorer"""
    from src.chat.semantic_interest.runtime_scorer import SemanticInterestScorer
    
    scorer = SemanticInterestScorer(model_path, use_fast_scorer=True)
    scorer.load()
    
    # 预热
    for text in SAMPLE_TEXTS[:3]:
        scorer.score(text)
    
    # 单条评分测试
    start = time.perf_counter()
    for _ in range(iterations):
        for text in SAMPLE_TEXTS:
            scorer.score(text)
    single_time = time.perf_counter() - start
    total_single = iterations * len(SAMPLE_TEXTS)
    
    # 批量评分测试
    start = time.perf_counter()
    for _ in range(iterations):
        scorer.score_batch(SAMPLE_TEXTS)
    batch_time = time.perf_counter() - start
    total_batch = iterations * len(SAMPLE_TEXTS)
    
    return {
        "mode": "fast_scorer",
        "single_total_time": single_time,
        "single_avg_ms": single_time / total_single * 1000,
        "single_qps": total_single / single_time,
        "batch_total_time": batch_time,
        "batch_avg_ms": batch_time / total_batch * 1000,
        "batch_qps": total_batch / batch_time,
    }


async def benchmark_async_scoring(model_path: str, iterations: int = 100):
    """测试异步评分"""
    from src.chat.semantic_interest.runtime_scorer import get_semantic_scorer
    
    scorer = await get_semantic_scorer(model_path, use_async=True)
    
    # 预热
    for text in SAMPLE_TEXTS[:3]:
        await scorer.score_async(text)
    
    # 单条异步评分
    start = time.perf_counter()
    for _ in range(iterations):
        for text in SAMPLE_TEXTS:
            await scorer.score_async(text)
    single_time = time.perf_counter() - start
    total_single = iterations * len(SAMPLE_TEXTS)
    
    # 并发评分（模拟高并发场景）
    start = time.perf_counter()
    for _ in range(iterations):
        tasks = [scorer.score_async(text) for text in SAMPLE_TEXTS]
        await asyncio.gather(*tasks)
    concurrent_time = time.perf_counter() - start
    total_concurrent = iterations * len(SAMPLE_TEXTS)
    
    return {
        "mode": "async",
        "single_total_time": single_time,
        "single_avg_ms": single_time / total_single * 1000,
        "single_qps": total_single / single_time,
        "concurrent_total_time": concurrent_time,
        "concurrent_avg_ms": concurrent_time / total_concurrent * 1000,
        "concurrent_qps": total_concurrent / concurrent_time,
    }


async def benchmark_batch_queue(model_path: str, iterations: int = 100):
    """测试批处理队列"""
    from src.chat.semantic_interest.optimized_scorer import get_fast_scorer
    
    queue = await get_fast_scorer(
        model_path,
        use_batch_queue=True,
        batch_size=8,
        flush_interval_ms=20.0
    )
    
    # 预热
    for text in SAMPLE_TEXTS[:3]:
        await queue.score(text)
    
    # 并发提交评分请求
    start = time.perf_counter()
    for _ in range(iterations):
        tasks = [queue.score(text) for text in SAMPLE_TEXTS]
        await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start
    total_requests = iterations * len(SAMPLE_TEXTS)
    
    stats = queue.get_statistics()
    
    await queue.stop()
    
    return {
        "mode": "batch_queue",
        "total_time": total_time,
        "avg_ms": total_time / total_requests * 1000,
        "qps": total_requests / total_time,
        "total_batches": stats["total_batches"],
        "avg_batch_size": stats["avg_batch_size"],
    }


def print_results(results: dict):
    """打印测试结果"""
    print(f"\n{'='*60}")
    print(f"模式: {results['mode']}")
    print(f"{'='*60}")
    
    if "single_avg_ms" in results:
        print(f"单条评分: {results['single_avg_ms']:.3f} ms/条, QPS: {results['single_qps']:.1f}")
    
    if "batch_avg_ms" in results:
        print(f"批量评分: {results['batch_avg_ms']:.3f} ms/条, QPS: {results['batch_qps']:.1f}")
    
    if "concurrent_avg_ms" in results:
        print(f"并发评分: {results['concurrent_avg_ms']:.3f} ms/条, QPS: {results['concurrent_qps']:.1f}")
    
    if "total_batches" in results:
        print(f"批处理队列: {results['avg_ms']:.3f} ms/条, QPS: {results['qps']:.1f}")
        print(f"  总批次: {results['total_batches']}, 平均批大小: {results['avg_batch_size']:.1f}")


async def main():
    """运行性能测试"""
    import sys
    
    # 检查模型路径
    model_dir = Path("data/semantic_interest/models")
    model_files = list(model_dir.glob("semantic_interest_*.pkl"))
    
    if not model_files:
        print("错误: 未找到模型文件，请先训练模型")
        print(f"模型目录: {model_dir}")
        sys.exit(1)
    
    # 使用最新的模型
    model_path = str(max(model_files, key=lambda p: p.stat().st_mtime))
    print(f"使用模型: {model_path}")
    
    iterations = 50  # 测试迭代次数
    
    print(f"\n测试配置: {iterations} 次迭代, {len(SAMPLE_TEXTS)} 条样本/次")
    print(f"总评分次数: {iterations * len(SAMPLE_TEXTS)} 条")
    
    # 1. sklearn 原始路径
    print("\n[1/4] 测试 sklearn 原始路径...")
    try:
        sklearn_results = benchmark_sklearn_scorer(model_path, iterations)
        print_results(sklearn_results)
    except Exception as e:
        print(f"sklearn 测试失败: {e}")
    
    # 2. FastScorer
    print("\n[2/4] 测试 FastScorer...")
    try:
        fast_results = benchmark_fast_scorer(model_path, iterations)
        print_results(fast_results)
    except Exception as e:
        print(f"FastScorer 测试失败: {e}")
    
    # 3. 异步评分
    print("\n[3/4] 测试异步评分...")
    try:
        async_results = await benchmark_async_scoring(model_path, iterations)
        print_results(async_results)
    except Exception as e:
        print(f"异步测试失败: {e}")
    
    # 4. 批处理队列
    print("\n[4/4] 测试批处理队列...")
    try:
        queue_results = await benchmark_batch_queue(model_path, iterations)
        print_results(queue_results)
    except Exception as e:
        print(f"批处理队列测试失败: {e}")
    
    # 性能对比总结
    print(f"\n{'='*60}")
    print("性能对比总结")
    print(f"{'='*60}")
    
    try:
        speedup = sklearn_results["single_avg_ms"] / fast_results["single_avg_ms"]
        print(f"FastScorer vs sklearn 单条: {speedup:.2f}x 加速")
        
        speedup = sklearn_results["batch_avg_ms"] / fast_results["batch_avg_ms"]
        print(f"FastScorer vs sklearn 批量: {speedup:.2f}x 加速")
    except:
        pass
    
    print("\n清理资源...")
    from src.chat.semantic_interest.optimized_scorer import shutdown_global_executor, clear_fast_scorer_instances
    from src.chat.semantic_interest.runtime_scorer import clear_scorer_instances
    
    shutdown_global_executor()
    clear_fast_scorer_instances()
    clear_scorer_instances()
    
    print("测试完成!")


if __name__ == "__main__":
    asyncio.run(main())
