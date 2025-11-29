"""沙盒危险模式检测测试"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.plugin_system.core.sandbox_environment import SandboxEnvironment, SandboxConfig


async def test_dangerous_patterns():
    """测试危险模式检测"""
    
    config = SandboxConfig(
        max_execution_time=5.0,
        max_memory_mb=128,
        max_cpu_time=5.0,
        allowed_modules=["json", "re", "math", "datetime", "random"]
    )
    
    sandbox = SandboxEnvironment(config)
    
    test_cases = [
        # (代码, 测试名称, 是否应该被阻止)
        ('result = "test".__class__.__name__', "访问 __class__", True),
        ('result = [].__class__.__bases__', "访问 __bases__", True),
        ('result = object.__subclasses__()', "访问 __subclasses__", True),
        ('result = str.__mro__', "访问 __mro__", True),
        ('result = (lambda: None).__globals__', "访问 __globals__", True),
        ('result = (lambda: None).__code__', "访问 __code__", True),
        ('result = __builtins__', "访问 __builtins__", True),
        ('result = 1 + 1', "正常运算", False),
        ('import json; result = json.dumps({"key": "value"})', "正常导入", False),
    ]
    
    print("="*60)
    print("沙盒危险模式检测测试")
    print("="*60 + "\n")
    
    passed = 0
    failed = 0
    
    for code, test_name, should_block in test_cases:
        print(f"测试: {test_name}")
        print(f"  代码: {code}")
        print(f"  预期: {'应该被阻止' if should_block else '应该通过'}")
        
        result = await sandbox.execute_async(code)
        
        is_blocked = not result["success"] and "禁止" in result.get("error", "")
        
        if should_block:
            if is_blocked:
                print(f"  结果: ✅ 成功阻止")
                passed += 1
            else:
                print(f"  结果: ❌ 未能阻止")
                print(f"  详情: {result}")
                failed += 1
        else:
            if not is_blocked and result["success"]:
                print(f"  结果: ✅ 正常执行")
                passed += 1
            else:
                print(f"  结果: ❌ 错误阻止")
                print(f"  详情: {result}")
                failed += 1
        
        print()
    
    total = passed + failed
    print("="*60)
    print(f"总测试数: {total}")
    print(f"✅ 通过: {passed} ({passed/total*100:.1f}%)")
    print(f"❌ 失败: {failed}")
    print("="*60)
    
    if failed == 0:
        print("🎉 所有测试通过! 危险模式检测工作正常!")
    else:
        print("⚠️  存在问题! 请检查失败的测试!")


if __name__ == "__main__":
    asyncio.run(test_dangerous_patterns())
