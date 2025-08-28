from enum import Enum

class NapcatEvent(Enum):
    # napcat插件事件枚举类 
    class ON_RECEIVED(Enum):
        TEXT = "napcat_on_received_text"  # 接收到文本消息
        FACE = "napcat_on_received_face"  # 接收到表情消息
        REPLY = "napcat_on_received_reply"  # 接收到回复消息
        IMAGE = "napcat_on_received_image"  # 接收到图像消息
        RECORD = "napcat_on_received_record"  # 接收到语音消息
        VIDEO = "napcat_on_received_video"  # 接收到视频消息
        AT = "napcat_on_received_at"  # 接收到at消息
        DICE = "napcat_on_received_dice"  # 接收到骰子消息
        SHAKE = "napcat_on_received_shake"  # 接收到屏幕抖动消息
        JSON = "napcat_on_received_json"  # 接收到JSON消息
        RPS = "napcat_on_received_rps"  # 接收到魔法猜拳消息
        FRIEND_INPUT = "napcat_on_friend_input"  # 好友正在输入
