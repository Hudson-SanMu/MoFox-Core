"""
沙盒安全测试脚本

测试沙盒环境的安全性，确保各种潜在的危险操作都被正确阻止
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.plugin_system.core.sandbox_environment import (
    SandboxConfig,
    SandboxEnvironment,
    SandboxTimeoutError,
    SandboxMemoryError,
    SandboxSecurityError,
)


class SecurityTestSuite:
    """沙盒安全测试套件"""

    def __init__(self):
        self.sandbox = SandboxEnvironment(SandboxConfig())
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []

    async def run_test(self, test_name: str, code: str, should_fail: bool = True, expected_error: str = None):
        """运行单个测试

        Args:
            test_name: 测试名称
            code: 要测试的代码
            should_fail: 是否应该失败（True表示代码应该被阻止）
            expected_error: 期望的错误类型
        """
        print(f"\n{'='*60}")
        print(f"测试: {test_name}")
        print(f"{'='*60}")
        print(f"代码:\n{code[:200]}{'...' if len(code) > 200 else ''}")
        print(f"预期: {'应该被阻止' if should_fail else '应该成功执行'}")

        try:
            result = await self.sandbox.execute_async(code, timeout=5.0)

            if result["success"]:
                if should_fail:
                    print(f"❌ 失败: 危险代码未被阻止!")
                    print(f"   执行结果: {result.get('result', 'None')}")
                    self.failed_tests += 1
                    self.test_results.append((test_name, "FAIL", "危险代码未被阻止"))
                else:
                    print(f"✅ 通过: 代码正常执行")
                    print(f"   执行结果: {result.get('result', 'None')}")
                    self.passed_tests += 1
                    self.test_results.append((test_name, "PASS", "代码正常执行"))
            else:
                error_type = result.get("error_type")
                error_msg = result.get("error")

                if should_fail:
                    if expected_error and expected_error not in error_type:
                        print(f"⚠️  警告: 被阻止但错误类型不符")
                        print(f"   期望错误: {expected_error}")
                        print(f"   实际错误: {error_type} - {error_msg}")
                        self.passed_tests += 1
                        self.test_results.append((test_name, "PASS*", f"被阻止 ({error_type})"))
                    else:
                        print(f"✅ 通过: 危险代码被成功阻止")
                        print(f"   错误类型: {error_type}")
                        print(f"   错误信息: {error_msg[:100]}{'...' if len(error_msg) > 100 else ''}")
                        self.passed_tests += 1
                        self.test_results.append((test_name, "PASS", f"成功阻止 ({error_type})"))
                else:
                    print(f"❌ 失败: 合法代码被错误阻止")
                    print(f"   错误类型: {error_type}")
                    print(f"   错误信息: {error_msg}")
                    self.failed_tests += 1
                    self.test_results.append((test_name, "FAIL", f"误拦截 ({error_type})"))

        except Exception as e:
            print(f"❌ 测试异常: {e}")
            self.failed_tests += 1
            self.test_results.append((test_name, "ERROR", str(e)))

    async def run_all_tests(self):
        """运行所有安全测试"""
        print("\n" + "="*60)
        print("开始沙盒安全测试")
        print("="*60)

        # ==================== 危险操作测试 ====================
        print("\n\n### 第一部分: 危险操作测试 ###\n")

        # 测试 1: eval
        await self.run_test(
            "阻止 eval",
            """
result = eval("1 + 1")
__result__ = result
""",
            should_fail=True,
            expected_error="NameError"
        )

        # 测试 2: exec
        await self.run_test(
            "阻止 exec",
            """
exec("print('hello')")
__result__ = "executed"
""",
            should_fail=True,
            expected_error="NameError"
        )

        # 测试 3: compile
        await self.run_test(
            "阻止 compile",
            """
code = compile("1 + 1", "<string>", "eval")
__result__ = code
""",
            should_fail=True,
            expected_error="NameError"
        )

        # 测试 4: __import__
        await self.run_test(
            "阻止 __import__ 导入未授权模块",
            """
os = __import__('os')
__result__ = os.system('echo hello')
""",
            should_fail=True,
            expected_error="SandboxSecurityError"
        )

        # 测试 5: open
        await self.run_test(
            "阻止 open 文件操作",
            """
f = open('/etc/passwd', 'r')
content = f.read()
__result__ = content
""",
            should_fail=True,
            expected_error="NameError"
        )

        # 测试 6: getattr
        await self.run_test(
            "阻止 getattr",
            """
import sys
result = getattr(sys, 'exit')
__result__ = result
""",
            should_fail=True,
            expected_error="NameError"
        )

        # 测试 7: setattr
        await self.run_test(
            "阻止 setattr",
            """
class Obj:
    pass
obj = Obj()
setattr(obj, 'dangerous', True)
__result__ = obj.dangerous
""",
            should_fail=True,
            expected_error="NameError"
        )

        # 测试 8: delattr
        await self.run_test(
            "阻止 delattr",
            """
class Obj:
    attr = 1
obj = Obj()
delattr(obj, 'attr')
__result__ = "deleted"
""",
            should_fail=True,
            expected_error="NameError"
        )

        # 测试 9: globals
        await self.run_test(
            "阻止 globals",
            """
g = globals()
__result__ = g
""",
            should_fail=True,
            expected_error="NameError"
        )

        # 测试 10: locals
        await self.run_test(
            "阻止 locals",
            """
l = locals()
__result__ = l
""",
            should_fail=True,
            expected_error="NameError"
        )

        # ==================== 模块导入测试 ====================
        print("\n\n### 第二部分: 模块导入测试 ###\n")

        # 测试 11: 导入 os
        await self.run_test(
            "阻止导入 os 模块",
            """
import os
__result__ = os.getcwd()
""",
            should_fail=True,
            expected_error="SandboxSecurityError"
        )

        # 测试 12: 导入 subprocess
        await self.run_test(
            "阻止导入 subprocess 模块",
            """
import subprocess
__result__ = subprocess.run(['ls'])
""",
            should_fail=True,
            expected_error="SandboxSecurityError"
        )

        # 测试 13: 导入 sys
        await self.run_test(
            "阻止导入 sys 模块",
            """
import sys
__result__ = sys.exit
""",
            should_fail=True,
            expected_error="SandboxSecurityError"
        )

        # 测试 14: 导入 socket
        await self.run_test(
            "阻止导入 socket 模块",
            """
import socket
s = socket.socket()
__result__ = "socket created"
""",
            should_fail=True,
            expected_error="SandboxSecurityError"
        )

        # 测试 15: 导入 requests
        await self.run_test(
            "阻止导入 requests 模块",
            """
import requests
__result__ = requests.get('http://example.com')
""",
            should_fail=True,
            expected_error="SandboxSecurityError"
        )

        # ==================== 允许的操作测试 ====================
        print("\n\n### 第三部分: 允许的操作测试 ###\n")

        # 测试 16: 基本算术
        await self.run_test(
            "允许基本算术运算",
            """
result = 1 + 2 * 3
__result__ = result
""",
            should_fail=False
        )

        # 测试 17: 字符串操作
        await self.run_test(
            "允许字符串操作",
            """
s = "Hello, World!"
__result__ = s.upper()
""",
            should_fail=False
        )

        # 测试 18: 列表操作
        await self.run_test(
            "允许列表操作",
            """
lst = [1, 2, 3, 4, 5]
result = [x * 2 for x in lst if x > 2]
__result__ = result
""",
            should_fail=False
        )

        # 测试 19: 字典操作
        await self.run_test(
            "允许字典操作",
            """
d = {'a': 1, 'b': 2}
d['c'] = 3
__result__ = sum(d.values())
""",
            should_fail=False
        )

        # 测试 20: 导入允许的模块 - json
        await self.run_test(
            "允许导入 json 模块",
            """
import json
data = {'key': 'value'}
result = json.dumps(data)
__result__ = result
""",
            should_fail=False
        )

        # 测试 21: 导入允许的模块 - re
        await self.run_test(
            "允许导入 re 模块",
            """
import re
pattern = r'\\d+'
text = "abc123def456"
result = re.findall(pattern, text)
__result__ = result
""",
            should_fail=False
        )

        # 测试 22: 导入允许的模块 - math
        await self.run_test(
            "允许导入 math 模块",
            """
import math
result = math.sqrt(16)
__result__ = result
""",
            should_fail=False
        )

        # 测试 23: 导入允许的模块 - datetime
        await self.run_test(
            "允许导入 datetime 模块",
            """
import datetime
now = datetime.datetime.now()
__result__ = now.year
""",
            should_fail=False
        )

        # 测试 24: 导入允许的模块 - random
        await self.run_test(
            "允许导入 random 模块",
            """
import random
result = random.randint(1, 100)
__result__ = result
""",
            should_fail=False
        )

        # ==================== 资源限制测试 ====================
        print("\n\n### 第四部分: 资源限制测试 ###\n")

        # 测试 25: 超时测试
        await self.run_test(
            "阻止超时代码",
            """
import time
time.sleep(10)  # 超过5秒超时限制
__result__ = "completed"
""",
            should_fail=True,
            expected_error="SandboxTimeoutError"
        )

        # 测试 26: 无限循环（应该超时）
        await self.run_test(
            "阻止无限循环",
            """
while True:
    pass
__result__ = "never reach here"
""",
            should_fail=True,
            expected_error="SandboxTimeoutError"
        )

        # ==================== 高级攻击测试 ====================
        print("\n\n### 第五部分: 高级攻击测试 ###\n")

        # 测试 27: 通过 __builtins__ 访问
        await self.run_test(
            "阻止通过 __builtins__ 访问危险函数",
            """
eval_func = __builtins__['eval']
result = eval_func("1 + 1")
__result__ = result
""",
            should_fail=True,
            expected_error="TypeError"  # __builtins__ 是字典，没有 eval
        )

        # 测试 28: 通过类的 __class__ 访问
        await self.run_test(
            "阻止通过 __class__ 访问",
            """
s = "test"
cls = s.__class__
__result__ = cls.__name__
""",
            should_fail=True,
            expected_error="AttributeError"
        )

        # 测试 29: 尝试修改 __builtins__
        await self.run_test(
            "阻止修改 __builtins__",
            """
__builtins__['eval'] = lambda x: x
__result__ = "modified"
""",
            should_fail=True,
            expected_error="TypeError"
        )

        # 测试 30: 递归炸弹
        await self.run_test(
            "阻止递归炸弹",
            """
def recursive():
    return recursive()
recursive()
__result__ = "never"
""",
            should_fail=True,
            expected_error="RecursionError"
        )

        # ==================== 打印测试结果 ====================
        self.print_summary()

    def print_summary(self):
        """打印测试摘要"""
        print("\n\n" + "="*60)
        print("测试摘要")
        print("="*60)

        total_tests = self.passed_tests + self.failed_tests
        pass_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0

        print(f"\n总测试数: {total_tests}")
        print(f"通过: {self.passed_tests} ({pass_rate:.1f}%)")
        print(f"失败: {self.failed_tests}")

        if self.failed_tests > 0:
            print("\n❌ 失败的测试:")
            for name, status, reason in self.test_results:
                if status == "FAIL":
                    print(f"  - {name}: {reason}")

        print("\n详细结果:")
        for name, status, reason in self.test_results:
            icon = "✅" if status.startswith("PASS") else "❌" if status == "FAIL" else "⚠️"
            print(f"{icon} {name}: {reason}")

        print("\n" + "="*60)
        if self.failed_tests == 0:
            print("🎉 所有测试通过! 沙盒环境安全可靠!")
        else:
            print("⚠️  存在安全风险! 请修复失败的测试!")
        print("="*60 + "\n")


async def main():
    """主测试函数"""
    test_suite = SecurityTestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MoFox-Bot 沙盒安全测试")
    print("="*60)
    print("\n测试目标: 验证沙盒环境能够阻止所有危险操作")
    print("测试范围: 危险函数、模块导入、资源限制、高级攻击")
    print("\n开始测试...\n")

    asyncio.run(main())
