from enum import Enum

class NapcatEvent(Enum):
    """
    napcat插件事件枚举类
    """
    class ON_RECEIVED(Enum): 
        """
        该分类下均为消息接受事件，只能由napcat_plugin触发
        """
        TEXT = "napcat_on_received_text"    
        '''接收到文本消息'''
        FACE = "napcat_on_received_face"    
        '''接收到表情消息'''
        REPLY = "napcat_on_received_reply"  
        '''接收到回复消息'''
        IMAGE = "napcat_on_received_image"  
        '''接收到图像消息'''
        RECORD = "napcat_on_received_record"    
        '''接收到语音消息'''
        VIDEO = "napcat_on_received_video"  
        '''接收到视频消息'''
        AT = "napcat_on_received_at"    
        '''接收到at消息'''
        DICE = "napcat_on_received_dice"    
        '''接收到骰子消息'''
        SHAKE = "napcat_on_received_shake"  
        '''接收到屏幕抖动消息'''
        JSON = "napcat_on_received_json"    
        '''接收到JSON消息'''
        RPS = "napcat_on_received_rps"  
        '''接收到魔法猜拳消息'''
        FRIEND_INPUT = "napcat_on_friend_input" 
        '''好友正在输入'''
    
    class ACCOUNT(Enum):
        """
        该分类是对账户相关的操作，只能由外部触发，napcat_plugin负责处理
        """
        SET_PROFILE = "napcat_set_qq_profile"   
        '''设置账号信息
        
        Args:
            nickname (Optional[str]): 名称(必须)
            personal_note (Optional[str]): 个性签名
            sex (Optional['0'|'1'|'2']): 性别
            raw (Optional[dict]): 原始请求体

        Returns:
            dict:  {
            "status": "ok",
            "retcode": 0,
            "data": {
                "result": 0,
                "errMsg": "string"
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }

        '''
        GET_ONLINE_CLIENTS = "napcat_get_online_clients"    
        '''获取当前账号在线客户端列表'''
        SET_ONLINE_STATUS = "napcat_set_online_status" 
        '''设置在线状态'''
        GET_FRIENDS_WITH_CATEGORY = "napcat_get_friends_with_category" 
        '''获取好友分组列表'''
        SET_AVATAR = "napcat_set_qq_avatar" 
        '''设置头像'''
        SEND_LIKE = "napcat_send_like"  '''点赞'''
        SET_FRIEND_ADD_REQUEST = "napcat_set_friend_add_request"    
        '''处理好友请求'''
        SET_SELF_LONGNICK = "napcat_set_self_longnick"  
        '''设置个性签名'''
        GET_LOGIN_INFO = "napcat_get_login_info"  
        '''获取登录号信息'''
        GET_RECENT_CONTACT = "napcat_get_recent_contact"    
        '''最近消息列表'''
        GET_STRANGER_INFO = "napcat_get_stranger_info"  
        '''获取(指定)账号信息'''
        GET_FRIEND_LIST = "napcat_get_friend_list"  
        '''获取好友列表'''
        GET_PROFILE_LIKE = "napcat_get_profile_like"    
        '''获取点赞列表'''
        DELETE_FRIEND = "napcat_delete_friend"  
        '''删除好友'''
        GET_USER_STATUS = "napcat_get_user_status"  
        '''获取用户状态'''
        GET_STATUS = "napcat_get_status"    
        '''获取状态'''
        GET_MINI_APP_ARK = "napcat_get_mini_app_ark"    
        '''获取小程序卡片'''
        SET_DIY_ONLINE_STATUS = "napcat_set_diy_online_status"  
        '''设置自定义在线状态'''
    
    class MESSAGE(Enum):
        """
        该分类是对信息相关的操作，只能由外部触发，napcat_plugin负责处理
        """
        SEND_GROUP_POKE = "napcat_send_group_poke"  
        '''发送群聊戳一戳'''
        SEND_PRIVATE_MSG = "napcat_send_private_msg"    
        '''发送私聊消息'''
        SEND_POKE = "napcat_send_friend_poke"    
        '''发送戳一戳'''
        DELETE_MSG = "napcat_delete_msg"    
        '''撤回消息'''
        GET_GROUP_MSG_HISTORY = "napcat_get_group_msg_history"  
        '''获取群历史消息'''
        GET_MSG = "napcat_get_msg"  
        '''获取消息详情'''
        GET_FORWARD_MSG = "napcat_get_forward_msg"  
        '''获取合并转发消息'''
        SET_MSG_EMOJI_LIKE = "napcat_set_msg_emoji_like"    
        '''贴表情'''
        GET_FRIEND_MSG_HISTORY = "napcat_get_friend_msg_history"    
        '''获取好友历史消息'''
        FETCH_EMOJI_LIKE = "napcat_fetch_emoji_like"    
        '''获取贴表情详情'''
        SEND_FORWARF_MSG = "napcat_send_forward_msg"    
        '''发送合并转发消息'''
        GET_RECOED = "napcat_get_record"    
        '''获取语音消息详情'''
        SEND_GROUP_AI_RECORD = "napcat_send_group_ai_record"    
        '''发送群AI语音'''

    class GROUP(Enum):
        """
        该分类是对群聊相关的操作，只能由外部触发，napcat_plugin负责处理
        """
        SET_GROUP_SEARCH = "napcat_set_group_search"
        '''设置群搜索'''
        GET_GROUP_DETAIL_INFO = "napcat_get_group_detail_info"  
        '''获取群详细信息'''
        SET_GROUP_ADD_OPTION = "napcat_set_group_add_option"    
        '''设置群添加选项'''
        SET_GROUP_ROBOT_ADD_OPTION = "napcat_set_group_robot_add_option"    
        '''设置群机器人添加选项'''
        SET_GROUP_KICK_MEMBERS = "napcat_set_group_kick_members"    
        '''批量踢出群成员'''
        SET_GROUP_KICK = "napcat_set_group_kick"    
        '''群踢人'''
        GET_GROUP_SYSTEM_MSG = "napcat_get_group_system_msg"    
        '''获取群系统消息'''
        SET_GROUP_BAN = "napcat_set_group_ban"  
        '''群禁言'''
        GET_ESSENCE_MSG_LIST = "napcat_get_essence_msg_list"    
        '''获取群精华消息'''
        SET_GROUP_WHOLE_BAN = "napcat_set_group_whole_ban"  
        '''全体禁言'''
        SET_GROUP_PORTRAINT = "napcat_set_group_portrait"   
        '''设置群头像'''
        SET_GROUP_ADMIN = "napcat_set_group_admin"  
        '''设置群管理'''
        SET_GROUP_CARD = "napcat_group_card"    
        '''设置群成员名片'''
        SET_ESSENCE_MSG = "napcat_set_essence_msg"  
        '''设置群精华消息'''
        SET_GROUP_NAME = "napcat_set_group_name"    
        '''设置群名'''
        DELETE_ESSENCE_MSG = "napcat_delete_essence_msg"    
        '''删除群精华消息'''
        SET_GROUP_LEAVE = "napcat_set_group_leave"  
        '''退群'''
        SEND_GROUP_NOTICE = "napcat_group_notice"   
        '''发送群公告'''
        SET_GROUP_SPECIAL_TITLE = "napcat_set_group_special_title"  
        '''设置群头衔'''
        GET_GROUP_NOTICE = "napcat_get_group_notice"    
        '''获取群公告'''
        SET_GROUP_ADD_REQUEST = "napcat_set_group_add_request"  
        '''处理加群请求'''
        GET_GROUP_INFO = "napcat_get_group_info"    
        '''获取群信息'''
        GET_GROUP_LIST = "napcat_get_group_list"    
        '''获取群列表'''
        DELETE_GROUP_NOTICE = "napcat_del_group_notice"
        '''删除群公告'''
        GET_GROUP_MEMBER_INFO = "napcat_get_group_member_info"
        '''获取群成员信息'''
        GET_GROUP_MEMBER_LIST = "napcat_get_group_member_list"
        '''获取群成员列表'''
        GET_GROUP_HONOR_INFO = "napcat_get_group_honor_info"
        '''获取群荣誉'''
        GET_GROUP_INFO_EX = "napcat_get_group_info_ex"
        '''获取群信息ex'''   
        GET_GROUP_AT_ALL_REMAIN = "napcat_get_group_at_all_remain"
        '''获取群 @全体成员 剩余次数'''
        GET_GROUP_SHUT_LIST = "napcat_get_group_shut_list"
        '''获取群禁言列表'''
        GET_GROUP_IGNORED_NOTIFIES = "napcat_get_group_ignored_notifies"
        '''获取群过滤系统消息'''
        SET_GROUP_SIGN = "napcat_set_group_sign"
        '''群打卡'''


