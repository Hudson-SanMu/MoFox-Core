# 权限系统使用说明

## 概述

MoFox_Bot的权限系统提供了完整的权限管理功能，支持权限等级和权限节点配置。系统包含以下核心概念：

- **Master用户**：拥有最高权限，无视所有权限节点，在配置文件中设置
- **权限节点**：细粒度的权限控制单元，由插件自行创建和管理
- **权限管理**：统一的权限授权、撤销和查询功能

## 配置文件设置

在 `config/bot_config.toml` 中添加权限配置：

```toml
[permission] # 权限系统配置
# Master用户配置（拥有最高权限，无视所有权限节点）
# 格式：[[platform, user_id], ...]
master_users = [
    ["qq", "123456789"],  # QQ平台的Master用户
    ["qq", "987654321"],  # 可以配置多个Master用户
]
```

## 插件开发中使用权限系统

### 1. 注册权限节点

在插件的 `on_load()` 方法中注册权限节点：

```python
from src.plugin_system.apis.permission_api import permission_api

class MyPlugin(BasePlugin):
    def on_load(self):
        # 注册权限节点
        permission_api.register_permission_node(
            "plugin.myplugin.admin",     # 权限节点名称
            "我的插件管理员权限",           # 权限描述
            "myplugin",                  # 插件名称
            False                        # 默认是否授权（False=默认拒绝）
        )
        
        permission_api.register_permission_node(
            "plugin.myplugin.user",
            "我的插件用户权限",
            "myplugin",
            True  # 默认授权
        )
```

### 2. 使用权限装饰器

最简单的权限检查方式是使用装饰器：

```python
from src.plugin_system.utils.permission_decorators import require_permission, require_master

class MyCommand(BaseCommand):
    @require_permission("plugin.myplugin.admin")
    async def execute(self, message: Message, chat_stream: ChatStream, args: List[str]):
        await send_message(chat_stream, "你有管理员权限！")
    
    @require_master("只有Master可以执行此操作")
    async def master_only_function(self, message: Message, chat_stream: ChatStream):
        await send_message(chat_stream, "Master专用功能")
```

### 3. 手动权限检查

对于更复杂的权限逻辑，可以手动检查权限：

```python
from src.plugin_system.utils.permission_decorators import PermissionChecker

class MyCommand(BaseCommand):
    async def execute(self, message: Message, chat_stream: ChatStream, args: List[str]):
        # 检查是否为Master用户
        if PermissionChecker.is_master(chat_stream):
            await send_message(chat_stream, "Master用户可以执行所有操作")
            return
        
        # 检查特定权限
        if PermissionChecker.check_permission(chat_stream, "plugin.myplugin.read"):
            await send_message(chat_stream, "你可以读取数据")
        
        # 使用 ensure_permission 自动发送权限不足消息
        if await PermissionChecker.ensure_permission(chat_stream, "plugin.myplugin.write"):
            await send_message(chat_stream, "你可以写入数据")
```

### 4. 直接使用权限API

```python
from src.plugin_system.apis.permission_api import permission_api

# 检查权限
has_permission = permission_api.check_permission("qq", "123456", "plugin.myplugin.admin")

# 检查是否为Master
is_master = permission_api.is_master("qq", "123456")

# 授权用户
success = permission_api.grant_permission("qq", "123456", "plugin.myplugin.admin")

# 撤销权限
success = permission_api.revoke_permission("qq", "123456", "plugin.myplugin.admin")

# 获取用户的所有权限
permissions = permission_api.get_user_permissions("qq", "123456")

# 获取所有权限节点
all_nodes = permission_api.get_all_permission_nodes()

# 获取指定插件的权限节点
plugin_nodes = permission_api.get_plugin_permission_nodes("myplugin")
```

## 权限管理命令

系统提供了内置的权限管理命令，需要相应权限才能使用：

### 管理员命令（需要 `plugin.permission.manage` 权限）

```
# 授权用户权限
/permission grant @用户 plugin.example.admin
/permission grant 123456789 plugin.example.admin

# 撤销用户权限
/permission revoke @用户 plugin.example.admin
/permission revoke 123456789 plugin.example.admin
```

### 查看命令（需要 `plugin.permission.view` 权限）

```
# 查看用户权限列表
/permission list @用户
/permission list 123456789
/permission list  # 查看自己的权限

# 检查用户是否拥有权限
/permission check @用户 plugin.example.admin
/permission check 123456789 plugin.example.admin

# 查看权限节点列表
/permission nodes  # 查看所有权限节点
/permission nodes example_plugin  # 查看指定插件的权限节点
```

### 帮助命令

```
/permission help  # 显示帮助信息
```

## 权限节点命名规范

建议使用以下命名规范：

```
plugin.<插件名>.<功能类别>.<具体权限>
```

示例：
- `plugin.music.play` - 音乐插件播放权限
- `plugin.music.admin` - 音乐插件管理权限
- `plugin.game.user` - 游戏插件用户权限
- `plugin.game.room.create` - 游戏插件房间创建权限

## 权限系统数据库表

系统会自动创建以下数据库表：

1. **permission_nodes** - 存储权限节点信息
2. **user_permissions** - 存储用户权限授权记录

## 最佳实践

1. **细粒度权限**：为不同功能创建独立的权限节点
2. **默认权限设置**：谨慎设置默认权限，敏感操作应默认拒绝
3. **权限描述**：为每个权限节点提供清晰的描述
4. **Master用户**：只为真正的管理员分配Master权限
5. **权限检查**：在执行敏感操作前始终检查权限

## 示例插件

查看 `plugins/permission_example.py` 了解完整的权限系统使用示例。

## 故障排除

1. **权限检查失败**：确保权限节点已正确注册
2. **Master用户配置**：检查配置文件中的用户ID格式是否正确
3. **权限不生效**：重启机器人以重新加载配置
4. **数据库问题**：检查数据库连接和表结构是否正确
