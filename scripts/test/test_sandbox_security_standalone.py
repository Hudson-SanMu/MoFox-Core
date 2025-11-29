"""
沙盒安全测试脚本 - 独立版本

测试沙盒环境的安全性，确保各种潜在的危险操作都被正确阻止
不依赖全局配置，可独立运行
"""
import asyncio
import io
import sys
import time
from typing import Any, Dict, Optional


class SandboxConfig:
    """沙盒配置"""

    def __init__(
        self,
        max_execution_time: float = 30.0,
        max_memory_mb: int = 256,
        max_cpu_time: float = 10.0,
        allowed_modules: Optional[list[str]] = None,
    ):
        self.max_execution_time = max_execution_time
        self.max_memory_mb = max_memory_mb
        self.max_cpu_time = max_cpu_time
        self.allowed_modules = allowed_modules or [
            "json",
            "re",
            "datetime",
            "time",
            "math",
            "random",
            "collections",
            "itertools",
            "functools",
            "typing",
        ]


class SandboxSecurityError(Exception):
    """沙盒安全违规异常"""
    pass


class RestrictedImporter:
    """受限的导入器"""

    def __init__(self, allowed_modules: list[str]):
        self.allowed_modules = set(allowed_modules)
        self.original_import = __builtins__.__import__

    def __call__(self, name: str, *args, **kwargs):
        base_module = name.split(".")[0]
        if base_module not in self.allowed_modules:
            raise SandboxSecurityError(f"模块 '{name}' 不在允许的导入列表中")
        return self.original_import(name, *args, **kwargs)


class SimpleSandbox:
    """简化的沙盒环境（用于测试）"""

    def __init__(self, config: SandboxConfig):
        self.config = config

    def _create_restricted_globals(self) -> Dict[str, Any]:
        """创建受限的全局命名空间"""
        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }

        restricted_globals = {
            "__builtins__": safe_builtins,
            "__name__": "__sandbox__",
            "__doc__": None,
        }

        if self.config.allowed_modules:
            restricted_globals["__builtins__"]["__import__"] = RestrictedImporter(self.config.allowed_modules)

        return restricted_globals

    def _check_dangerous_patterns(self, code: str):
        """检查代码中的危险模式"""
        import re
        
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
                raise SandboxSecurityError(error_msg)

    async def execute_async(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """异步执行代码"""
        timeout = timeout or self.config.max_execution_time

        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._execute_sync, code, context),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"执行超时（{timeout}秒）",
                "error_type": "SandboxTimeoutError",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def _execute_sync(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """同步执行代码"""
        restricted_globals = self._create_restricted_globals()

        if context:
            for key, value in context.items():
                if not key.startswith("_"):
                    restricted_globals[key] = value

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()

        try:
            sys.stdout = output_buffer
            sys.stderr = error_buffer

            # 检查危险模式
            self._check_dangerous_patterns(code)

            compiled_code = compile(code, "<sandbox>", "exec")
            exec(compiled_code, restricted_globals)

            result_value = restricted_globals.get("__result__", None)

            return {
                "success": True,
                "result": result_value,
                "output": output_buffer.getvalue(),
            }

        except SandboxSecurityError as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "SandboxSecurityError",
                "output": output_buffer.getvalue(),
            }

        except Exception as e:
            error_output = error_buffer.getvalue()
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "error_traceback": error_output,
                "output": output_buffer.getvalue(),
            }

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class SecurityTestSuite:
    """沙盒安全测试套件"""

    def __init__(self):
        self.sandbox = SimpleSandbox(SandboxConfig())
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []

    async def run_test(self, test_name: str, code: str, should_fail: bool = True, expected_error: str = None):
        """运行单个测试"""
        print(f"\n{'='*60}")
        print(f"测试: {test_name}")
        print(f"{'='*60}")
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
                    self.passed_tests += 1
                    self.test_results.append((test_name, "PASS", "代码正常执行"))
            else:
                error_type = result.get("error_type")
                error_msg = result.get("error")

                if should_fail:
                    print(f"✅ 通过: 危险代码被成功阻止")
                    print(f"   错误类型: {error_type}")
                    print(f"   错误信息: {error_msg[:80]}...")
                    self.passed_tests += 1
                    self.test_results.append((test_name, "PASS", f"成功阻止 ({error_type})"))
                else:
                    print(f"❌ 失败: 合法代码被错误阻止")
                    print(f"   错误类型: {error_type}")
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

        print("\n### 第一部分: 危险函数测试 ###\n")

        await self.run_test("阻止 eval", 'result = eval("1+1")\n__result__ = result', True)
        await self.run_test("阻止 exec", 'exec("print(1)")\n__result__ = "ok"', True)
        await self.run_test("阻止 compile", 'compile("1+1", "<>", "eval")', True)
        await self.run_test("阻止 open", 'open("test.txt", "r")', True)
        await self.run_test("阻止 __import__", '__import__("os")', True)
        await self.run_test("阻止 getattr", 'getattr(str, "upper")', True)
        await self.run_test("阻止 setattr", 'class A: pass\nsetattr(A(), "x", 1)', True)
        await self.run_test("阻止 globals", 'globals()', True)
        await self.run_test("阻止 locals", 'locals()', True)

        print("\n### 第二部分: 模块导入测试 ###\n")

        await self.run_test("阻止导入 os", 'import os\n__result__ = os.getcwd()', True)
        await self.run_test("阻止导入 subprocess", 'import subprocess', True)
        await self.run_test("阻止导入 sys", 'import sys', True)
        await self.run_test("阻止导入 socket", 'import socket', True)

        print("\n### 第三部分: 允许的操作测试 ###\n")

        await self.run_test("允许基本算术", '__result__ = 1 + 2 * 3', False)
        await self.run_test("允许字符串", '__result__ = "Hello".upper()', False)
        await self.run_test("允许列表", '__result__ = [x*2 for x in [1,2,3]]', False)
        await self.run_test("允许字典", '__result__ = {"a": 1, "b": 2}', False)
        await self.run_test("允许 json", 'import json\n__result__ = json.dumps({"a":1})', False)
        await self.run_test("允许 re", 'import re\n__result__ = re.findall(r"\\d+", "a1b2")', False)
        await self.run_test("允许 math", 'import math\n__result__ = math.sqrt(16)', False)
        await self.run_test("允许 datetime", 'import datetime\n__result__ = datetime.datetime.now().year', False)
        await self.run_test("允许 random", 'import random\n__result__ = random.randint(1,10)', False)

        print("\n### 第四部分: 资源限制测试 ###\n")

        await self.run_test("阻止超时", 'import time\ntime.sleep(10)', True)
        await self.run_test("阻止无限循环", 'while True: pass', True)

        print("\n### 第五部分: 高级攻击测试 ###\n")

        await self.run_test("阻止通过 __builtins__ 访问", '__builtins__["eval"]("1+1")', True)
        await self.run_test("阻止通过 __class__ 访问", '"test".__class__.__name__', True)
        await self.run_test("阻止递归炸弹", 'def f(): return f()\nf()', True)

        self.print_summary()

    def print_summary(self):
        """打印测试摘要"""
        print("\n\n" + "="*60)
        print("测试摘要")
        print("="*60)

        total_tests = self.passed_tests + self.failed_tests
        pass_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0

        print(f"\n总测试数: {total_tests}")
        print(f"✅ 通过: {self.passed_tests} ({pass_rate:.1f}%)")
        print(f"❌ 失败: {self.failed_tests}")

        if self.failed_tests > 0:
            print("\n失败的测试:")
            for name, status, reason in self.test_results:
                if status == "FAIL":
                    print(f"  - {name}: {reason}")

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
