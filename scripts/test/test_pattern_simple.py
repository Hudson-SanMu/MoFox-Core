"""简单的危险模式检测测试"""
import re


def check_dangerous_patterns(code: str) -> tuple[bool, str]:
    """检查代码中的危险模式
    
    Returns:
        (是否安全, 错误消息)
    """
    dangerous_patterns = [
        (r"__class__", "禁止访问 __class__ 属性"),
        (r"__bases__", "禁止访问 __bases__ 属性"),
        (r"__subclasses__", "禁止访问 __subclasses__ 方法"),
        (r"__mro__", "禁止访问 __mro__ 属性"),
        (r"__globals__", "禁止访问 __globals__ 属性"),
        (r"__code__", "禁止访问 __code__ 属性"),
        (r"__builtins__", "禁止直接访问 __builtins__"),
        (r"func_globals", "禁止访问 func_globals"),
        (r"gi_frame", "禁止访问 gi_frame"),
        (r"gi_code", "禁止访问 gi_code"),
    ]
    
    for pattern, error_msg in dangerous_patterns:
        if re.search(pattern, code):
            return False, error_msg
    
    return True, ""


def main():
    """主测试函数"""
    
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
        ('def test_class(): pass', "定义函数（包含class关键字）", False),
        ('message = "Welcome to our class!"', "普通字符串（包含class单词）", False),
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
        
        is_safe, error_msg = check_dangerous_patterns(code)
        is_blocked = not is_safe
        
        if should_block:
            if is_blocked:
                print(f"  结果: ✅ 成功阻止 - {error_msg}")
                passed += 1
            else:
                print(f"  结果: ❌ 未能阻止 - 代码未被检测为危险")
                failed += 1
        else:
            if is_safe:
                print(f"  结果: ✅ 正常通过")
                passed += 1
            else:
                print(f"  结果: ❌ 错误阻止 - {error_msg}")
                failed += 1
        
        print()
    
    total = passed + failed
    print("="*60)
    print(f"总测试数: {total}")
    print(f"✅ 通过: {passed} ({passed/total*100:.1f}%)")
    print(f"❌ 失败: {failed}")
    print("="*60 + "\n")
    
    if failed == 0:
        print("🎉 所有测试通过! 危险模式检测工作正常!")
    else:
        print("⚠️  存在问题!")
        print("注意: 检测到 'class' 关键字在正常代码中也会被阻止")
        print("建议: 使用更精确的正则表达式，如 r'\\.__class__' 或 r'\\b__class__\\b'")


if __name__ == "__main__":
    main()
