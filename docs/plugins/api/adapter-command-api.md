# 适配器命令API

这个API允许插件向适配器发送命令并获取返回值，可以用于获取群列表、好友列表等功能。

## 主要功能

### 向适配器发送命令并获取返回值

```python
async def adapter_command_to_stream(
    action: str,
    params: dict,
    stream_id: Optional[str] = None,
    timeout: float = 30.0,
    storage_message: bool = False,
    show_log: bool = True,
) -> dict:
```

向适配器发送命令并等待响应。

**Args:**
- `action` (str): 适配器命令动作，如"get_group_list"、"get_friend_list"等
- `params` (dict): 命令参数字典
- `stream_id` (Optional[str]): 聊天流ID，可选，如果不提供则自动生成一个
- `timeout` (float): 超时时间（秒），默认30秒
- `storage_message` (bool): 是否存储消息到数据库，默认False
- `show_log` (bool): 是否显示日志，默认True


简化版的适配器命令调用，自动生成stream_id。

**Args:**
- `action` (str): 适配器命令动作
- `params` (dict): 命令参数字典，默认为空字典
- `timeout` (float): 超时时间（秒），默认30秒
- `show_log` (bool): 是否显示日志，默认True

**Returns:**
- `dict` - 适配器返回的响应，格式为:
  - 成功: `{"status": "ok", "data": {...}, "message": "..."}`
  - 失败: `{"status": "failed", "message": "错误信息"}`
  - 错误: `{"status": "error", "message": "错误信息"}`

## 使用示例

### 1. 获取群列表

```python
from src.plugin_system.apis import send_api

async def get_group_list(stream_id: str):
    """获取机器人加入的群列表"""
    
    response = await send_api.adapter_command_to_stream(
        action="get_group_list",
        params={},
        stream_id=stream_id
    )
    
    if response["status"] == "ok":
        group_list = response.get("data", [])
        logger.info(f"获取到 {len(group_list)} 个群")
        return group_list
    else:
        logger.error(f"获取群列表失败: {response['message']}")
        return []
```

### 2. 获取好友列表

```python
async def get_friend_list(stream_id: str):
    """获取机器人的好友列表"""
    
    response = await send_api.adapter_command_to_stream(
        action="get_friend_list",
        params={},
        stream_id=stream_id
    )
    
    if response["status"] == "ok":
        friend_list = response.get("data", [])
        logger.info(f"获取到 {len(friend_list)} 个好友")
        return friend_list
    else:
        logger.error(f"获取好友列表失败: {response['message']}")
        return []
```

### 3. 获取群成员信息

```python
async def get_group_member_info(stream_id: str, group_id: int, user_id: int):
    """获取群成员信息"""
    
    response = await send_api.adapter_command_to_stream(
        action="get_group_member_info",
        params={
            "group_id": group_id,
            "user_id": user_id
        },
        stream_id=stream_id
    )
    
    if response["status"] == "ok":
        member_info = response.get("data", {})
        logger.info(f"获取到群成员信息: {member_info}")
        return member_info
    else:
        logger.error(f"获取群成员信息失败: {response['message']}")
        return None
```

### 4. 在命令插件中使用

```python
from src.plugin_system.base.base_command import BaseCommand
from src.plugin_system.apis import send_api
from typing import Tuple

class GetGroupListCommand(BaseCommand):
    """获取群列表命令"""
    
    command_name = "get_groups"
    command_description = "获取机器人加入的群列表"
    command_pattern = r"^/get_groups$"
    command_help = "获取机器人加入的群列表"
    command_examples = ["/get_groups"]
    intercept_message = True
    
    async def execute(self) -> Tuple[bool, str, bool]:
        try:
            # 获取聊天流ID
            stream_id = self.message.chat_stream.stream_id
            
            # 调用适配器命令API
            response = await send_api.adapter_command_to_stream(
                action="get_group_list",
                params={},
                stream_id=stream_id
            )
            
            if response["status"] == "ok":
                group_list = response.get("data", [])
                
                if group_list:
                    # 格式化群列表信息
                    group_info = "\\n".join([
                        f"群号: {group['group_id']}, 群名: {group['group_name']}"
                        for group in group_list
                    ])
                    await self.send_text(f"机器人加入的群列表:\\n{group_info}")
                else:
                    await self.send_text("机器人未加入任何群")
                
                return True, "获取群列表成功", True
            else:
                await self.send_text(f"获取群列表失败: {response['message']}")
                return False, "获取群列表失败", True
                
        except Exception as e:
            logger.error(f"执行获取群列表命令时出错: {e}")
            await self.send_text("命令执行失败")
            return False, "命令执行失败", True
```

### 5. 带超时设置的调用

```python
async def get_group_info_with_timeout(stream_id: str, group_id: int):
    """获取群信息，设置较短的超时时间"""
    
    response = await send_api.adapter_command_to_stream(
        action="get_group_info",
        params={"group_id": group_id},
        stream_id=stream_id,
        timeout=10.0,  # 10秒超时
        show_log=True
    )
    
    if response["status"] == "ok":
        return response.get("data", {})
    elif response["status"] == "error" and "timeout" in response["message"]:
        logger.warning("获取群信息超时")
        return None
    else:
        logger.error(f"获取群信息失败: {response['message']}")
        return None
```

## 支持的适配器命令

以下是一些常用的适配器命令（具体支持的命令取决于适配器实现）：

- `get_group_list` - 获取群列表
- `get_friend_list` - 获取好友列表
- `get_group_info` - 获取群信息
- `get_group_member_list` - 获取群成员列表
- `get_group_member_info` - 获取群成员信息
- `get_stranger_info` - 获取陌生人信息
- `get_msg` - 获取消息
- `get_forward_msg` - 获取转发消息

## 注意事项

1. 所有命令都是异步执行的，需要使用`await`关键字
2. 超时时间建议根据命令复杂度设置，默认30秒
3. 命令失败时会返回错误信息，建议进行错误处理
4. 不建议在消息存储中记录适配器命令消息
5. 适配器命令的具体参数和返回值格式请参考NapCat文档

## 错误处理

```python
async def safe_adapter_command(stream_id: str, action: str, params: dict):
    """安全的适配器命令调用"""
    try:
        response = await send_api.adapter_command_to_stream(
            action=action,
            params=params,
            stream_id=stream_id,
            timeout=15.0
        )
        
        if response["status"] == "ok":
            return response.get("data")
        elif response["status"] == "error":
            if "timeout" in response["message"]:
                logger.warning(f"适配器命令 {action} 超时")
            else:
                logger.error(f"适配器命令 {action} 执行错误: {response['message']}")
        else:
            logger.error(f"适配器命令 {action} 执行失败: {response['message']}")
            
        return None
        
    except Exception as e:
        logger.error(f"调用适配器命令时出错: {e}")
        return None
```
