# 插件Python依赖管理系统

## 概述

插件系统现在支持自动检查和安装Python包依赖。当插件初始化时，系统会：

1. 检查插件所需的Python包是否已安装
2. 验证包版本是否满足要求
3. 自动安装缺失的依赖包（可配置）
4. 提供详细的错误信息和日志

## 配置依赖

### 方式1: 简单字符串列表（向后兼容）

```python
from src.plugin_system import BasePlugin

@register_plugin
class MyPlugin(BasePlugin):
    # 简单的字符串列表格式
    python_dependencies: List[str] = [
        "requests", 
        "beautifulsoup4>=4.9.0",
        "httpx[socks]"
    ]
```

### 方式2: 详细的PythonDependency对象（推荐）

```python
from src.plugin_system import BasePlugin, PythonDependency

@register_plugin
class MyPlugin(BasePlugin):
    python_dependencies: List[PythonDependency] = [
        PythonDependency(
            package_name="requests",
            version=">=2.25.0",
            description="HTTP请求库",
            optional=False
        ),
        PythonDependency(
            package_name="beautifulsoup4",
            version=">=4.9.0", 
            description="HTML解析库",
            optional=False
        ),
        PythonDependency(
            package_name="httpx",
            install_name="httpx[socks]",  # 安装时使用的名称
            description="支持SOCKS代理的HTTP客户端",
            optional=True
        )
    ]
```

## PythonDependency参数说明

- `package_name`: 包名称（用于import检查）
- `version`: 版本要求，支持PEP 440格式（如 ">=1.0.0", "==2.1.3"）
- `install_name`: pip安装时使用的名称（如果与package_name不同）
- `description`: 依赖描述，用于日志和错误信息
- `optional`: 是否为可选依赖，可选依赖缺失不会阻止插件加载

## 全局配置

创建 `mmc/config/dependency_config.toml` 文件来配置依赖管理行为：

```toml
[dependency_management]
# 是否启用自动安装（主开关）
auto_install = true

# 安装超时时间（秒）
auto_install_timeout = 300

# 是否使用PyPI镜像源（推荐，可加速下载）
use_mirror = true
mirror_url = "https://pypi.tuna.tsinghua.edu.cn/simple"

# 是否使用网络代理（高级选项）
use_proxy = false
proxy_url = ""

# pip安装选项
pip_options = [
    "--no-warn-script-location",
    "--disable-pip-version-check"
]

# 安装前是否提示用户
prompt_before_install = false

# 日志级别
install_log_level = "INFO"
```

## PyPI镜像源配置（推荐）

使用PyPI镜像源可以显著加速包的下载，特别是在中国大陆地区：

```toml
[dependency_management]
use_mirror = true
mirror_url = "https://pypi.tuna.tsinghua.edu.cn/simple"  # 清华大学镜像源
```

### 常用的国内镜像源

```toml
# 清华大学镜像源（推荐）
mirror_url = "https://pypi.tuna.tsinghua.edu.cn/simple"

# 阿里云镜像源
mirror_url = "https://mirrors.aliyun.com/pypi/simple"

# 中科大镜像源
mirror_url = "https://pypi.mirrors.ustc.edu.cn/simple"

# 豆瓣镜像源
mirror_url = "https://pypi.douban.com/simple"
```

## 网络代理配置（高级选项）

如果需要通过网络代理安装包，可以配置：

```toml
[dependency_management]
use_proxy = true
proxy_url = "http://proxy.example.com:8080"
```

**注意**：推荐优先使用PyPI镜像源而不是代理，镜像源通常更快更稳定。

## 编程方式配置

也可以通过代码动态配置依赖管理：

```python
from src.plugin_system.utils.dependency_config import configure_dependency_settings

# 禁用自动安装
configure_dependency_settings(auto_install=False)

# 设置PyPI镜像源（推荐）
configure_dependency_settings(
    use_mirror=True,
    mirror_url="https://pypi.tuna.tsinghua.edu.cn/simple"
)

# 设置网络代理
configure_dependency_settings(
    use_proxy=True,
    proxy_url="http://proxy.example.com:8080"
)

# 修改超时时间
configure_dependency_settings(auto_install_timeout=600)
```

**注意**：编程方式的配置更改不会持久化，实际配置请修改 `bot_config.toml` 文件。

## 工作流程

1. **插件初始化**: 当插件类被实例化时，系统自动检查依赖
2. **依赖标准化**: 将字符串格式的依赖转换为`PythonDependency`对象
3. **检查已安装**: 尝试导入每个依赖包并检查版本
4. **智能别名解析 (新增)**: 如果直接导入失败 (例如 `import beautifulsoup4` 失败)，系统会查询内置的别名映射表 (例如 `beautifulsoup4` -> `bs4`)，并尝试使用别名再次导入。
5. **自动安装**: 如果启用，自动安装缺失的依赖
6. **错误处理**: 记录详细的错误信息和安装日志

## 智能别名解析 (Smart Alias Resolution)

为了提升开发体验，依赖管理系统内置了一套智能别名解析机制。

### 解决的问题

Python生态中存在一些特殊的包，它们的**安装名** (在 `pip install` 中使用) 与**导入名** (在 `import` 语句中使用) 不一致。最典型的例子就是：
- 安装名: `beautifulsoup4`, 导入名: `bs4`
- 安装名: `Pillow`, 导入名: `PIL`
- 安装名: `scikit-learn`, 导入名: `sklearn`

如果开发者在 `python_dependencies` 列表中使用简单的字符串格式 `"beautifulsoup4"`，标准的依赖检查会因为无法 `import beautifulsoup4` 而失败。

### 工作原理

当依赖管理器通过包名直接导入失败时，它会：
1. 查询一个内置的、包含上百个常见包的别名映射表。
2. 如果在表中找到对应的导入名，则使用该别名再次尝试导入。
3. 如果使用别名导入成功，则依赖检查通过，并继续进行版本验证。

这个过程是自动的，旨在处理绝大多数常见情况，减少开发者手动配置的麻烦。

### 注意事项

- **最佳实践**: 尽管有智能别名解析，我们仍然**强烈推荐**使用 `PythonDependency` 对象来明确指定 `package_name` (导入名) 和 `install_name` (安装名)，这能确保最高的准确性和可读性。
- **覆盖范围**: 内置的别名映射表涵盖了大量常用库，但无法保证100%覆盖所有情况。如果遇到别名库未收录的包，请使用 `PythonDependency` 对象进行精确定义。

## 日志输出示例

```
[Plugin:web_search_tool] 开始自动安装Python依赖: ['asyncddgs', 'httpx[socks]']
[Plugin:web_search_tool] 使用PyPI镜像源: https://pypi.tuna.tsinghua.edu.cn/simple
[Plugin:web_search_tool] ✅ 成功安装: asyncddgs
[Plugin:web_search_tool] ✅ 成功安装: httpx[socks]
[Plugin:web_search_tool] 🎉 所有依赖安装完成
[Plugin:web_search_tool] Python依赖检查通过
```

## 错误处理

当依赖检查失败时，系统会：

1. 记录详细的错误信息
2. 如果是可选依赖缺失，仅记录警告
3. 如果是必需依赖缺失且自动安装失败，阻止插件加载
4. 提供清晰的解决建议

## 最佳实践

1. **优先使用`PythonDependency`对象**: 这是最可靠、最明确的方式，尤其是在安装名和导入名不同时。
2. **利用智能别名解析**: 对于常见的、安装名与导入名不一致的包 (如 `beautifulsoup4`, `Pillow` 等)，可以直接在字符串列表里使用安装名，系统会自动解析。
3. **配置PyPI镜像源**: 特别是在中国大陆地区，可显著提升下载速度。
4. **合理设置可选依赖**: 避免非核心功能阻止插件加载。
5. **指定版本要求**: 确保兼容性。
6. **添加描述信息**: 帮助用户理解依赖的用途。
7. **测试依赖配置**: 在不同环境中验证依赖是否正确。

## 安全考虑

- 自动安装功能默认启用，但可以通过`auto_install=false`配置禁用
- 所有安装操作都有详细的日志记录
- 支持设置安装超时以避免长时间挂起
- 通过单一的`auto_install`开关控制所有自动安装行为

## 故障排除

### 依赖安装失败

1. **优先尝试PyPI镜像源**：配置国内镜像源如清华、阿里云等
2. 检查网络连接
3. 验证镜像源URL是否正确
4. 如果镜像源失败，可尝试禁用镜像源或更换其他镜像源
5. 检查代理设置（如果使用）
6. 查看详细的错误日志

### 版本冲突

1. 检查现有包的版本
2. 调整版本要求
3. 考虑使用虚拟环境

### 导入错误

1. **确认包名与导入名**: 检查安装名和导入名是否一致。如果不一致，推荐使用 `PythonDependency` 对象明确指定 `package_name` 和 `install_name`。
2. **利用自动别名解析**: 对于常见库，系统会自动尝试解析别名。如果你的库比较冷门且名称不一致，请使用 `PythonDependency` 对象。
3. **检查可选依赖配置**: 确认 `optional=True` 是否被正确设置。
4. **验证安装是否成功**: 查看日志，确认 `pip install` 过程没有报错。

