from enum import Enum


class NapcatEvent:
    """
    napcat插件事件枚举类
    """

    class ON_RECEIVED(Enum):
        """
        该分类下均为消息接受事件，只能由napcat_plugin触发
        """

        TEXT = "napcat_on_received_text"
        """接收到文本消息"""
        FACE = "napcat_on_received_face"
        """接收到表情消息"""
        REPLY = "napcat_on_received_reply"
        """接收到回复消息"""
        IMAGE = "napcat_on_received_image"
        """接收到图像消息"""
        RECORD = "napcat_on_received_record"
        """接收到语音消息"""
        VIDEO = "napcat_on_received_video"
        """接收到视频消息"""
        AT = "napcat_on_received_at"
        """接收到at消息"""
        DICE = "napcat_on_received_dice"
        """接收到骰子消息"""
        SHAKE = "napcat_on_received_shake"
        """接收到屏幕抖动消息"""
        JSON = "napcat_on_received_json"
        """接收到JSON消息"""
        RPS = "napcat_on_received_rps"
        """接收到魔法猜拳消息"""
        FRIEND_INPUT = "napcat_on_friend_input"
        """好友正在输入"""
        EMOJI_LIEK = "napcat_on_received_emoji_like"
        """接收到群聊表情回复"""

    class ACCOUNT(Enum):
        """
        该分类是对账户相关的操作，只能由外部触发，napcat_plugin负责处理
        """

        SET_PROFILE = "napcat_set_qq_profile"
        """设置账号信息
        
        Args:
            nickname (Optional[str]): 名称(必须)
            personal_note (Optional[str]): 个性签名
            sex ('0'|'1'|'2'): 性别
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

        """
        GET_ONLINE_CLIENTS = "napcat_get_online_clients"
        """获取当前账号在线客户端列表
        
        Args:
            no_cache (Optional[bool]):  是否不使用缓存
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": [
                "string"
            ],
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SET_ONLINE_STATUS = "napcat_set_online_status"
        """设置在线状态
        
        Args:
            status (Optional[str]): 状态代码(必须)
            ext_status (Optional[str]): 额外状态代码,默认为0
            battery_status (Optional[str]): 电池信息,默认为0
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_FRIENDS_WITH_CATEGORY = "napcat_get_friends_with_category"
        """获取好友分组列表
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": [
                {
                    "categoryId": 0,
                    "categorySortId": 0,
                    "categoryName": "string",
                    "categoryMbCount": 0,
                    "onlineCount": 0,
                    "buddyList": [
                        {
                            "birthday_year": 0,
                            "birthday_month": 0,
                            "birthday_day": 0,
                            "user_id": 0,
                            "age": 0,
                            "phone_num": "string",
                            "email": "string",
                            "category_id": 0,
                            "nickname": "string",
                            "remark": "string",
                            "sex": "string",
                            "level": 0
                        }
                    ]
                }
            ],
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SET_AVATAR = "napcat_set_qq_avatar"
        """设置头像
        
        Args:
            file (Optional[str]): 文件路径或base64(必需)
            raw (Optional[dict]): 原始请求体

        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SEND_LIKE = "napcat_send_like"
        """点赞
        
        Args:
            user_id (Optional[str|int]): 用户id(必需)
            times (Optional[int]): 点赞次数,默认1
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SET_FRIEND_ADD_REQUEST = "napcat_set_friend_add_request"
        """处理好友请求
        
        Args:
            flag (Optional[str]): 请求id(必需)
            approve (Optional[bool]): 是否同意(必需)
            remark (Optional[str]): 好友备注(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SET_SELF_LONGNICK = "napcat_set_self_longnick"
        """设置个性签名
        
        Args:
            longNick (Optional[str]): 内容(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
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
        """
        GET_LOGIN_INFO = "napcat_get_login_info"
        """获取登录号信息
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "user_id": 0,
                "nickname": "string"
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_RECENT_CONTACT = "napcat_get_recent_contact"
        """最近消息列表

        Args:
            count (Optional[int]): 会话数量
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": [
                {
                    "lastestMsg": {
                        "self_id": 0,
                        "user_id": 0,
                        "time": 0,
                        "real_seq": "string",
                        "message_type": "string",
                        "sender": {
                            "user_id": 0,
                            "nickname": "string",
                            "sex": "male",
                            "age": 0,
                            "card": "string",
                            "role": "owner"
                        },
                        "raw_message": "string",
                        "font": 0,
                        "sub_type": "string",
                        "message": [
                            {
                                "type": "text",
                                "data": {
                                    "text": "string"
                                }
                            }
                        ],
                        "message_format": "string",
                        "post_type": "string",
                        "group_id": 0
                    },
                    "peerUin": "string",
                    "remark": "string",
                    "msgTime": "string",
                    "chatType": 0,
                    "msgId": "string",
                    "sendNickName": "string",
                    "sendMemberName": "string",
                    "peerName": "string"
                }
            ],
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_STRANGER_INFO = "napcat_get_stranger_info"
        """获取(指定)账号信息
        
        Args:
            user_id (Optional[str|int]): 用户id(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "user_id": 0,
                "uid": "string",
                "uin": "string",
                "nickname": "string",
                "age": 0,
                "qid": "string",
                "qqLevel": 0,
                "sex": "string",
                "long_nick": "string",
                "reg_time": 0,
                "is_vip": true,
                "is_years_vip": true,
                "vip_level": 0,
                "remark": "string",
                "status": 0,
                "login_days": 0
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_FRIEND_LIST = "napcat_get_friend_list"
        """获取好友列表
        
        Args:
            no_cache (Optional[bool]): 是否不使用缓存
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": [
                {
                    "birthday_year": 0,
                    "birthday_month": 0,
                    "birthday_day": 0,
                    "user_id": 0,
                    "age": 0,
                    "phone_num": "string",
                    "email": "string",
                    "category_id": 0,
                    "nickname": "string",
                    "remark": "string",
                    "sex": "string",
                    "level": 0
                }
            ],
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_PROFILE_LIKE = "napcat_get_profile_like"
        """获取点赞列表
        
        Args:
            user_id (Optional[str|int]): 用户id,指定用户,不填为获取所有
            start (Optional[int]): 起始值
            count (Optional[int]): 返回数量
            raw (Optional[dict]): 原始请求体

        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "uid": "string",
                "time": 0,
                "favoriteInfo": {
                    "total_count": 0,
                    "last_time": 0,
                    "today_count": 0,
                    "userInfos": [
                        {
                            "uid": "string",
                            "src": 0,
                            "latestTime": 0,
                            "count": 0,
                            "giftCount": 0,
                            "customId": 0,
                            "lastCharged": 0,
                            "bAvailableCnt": 0,
                            "bTodayVotedCnt": 0,
                            "nick": "string",
                            "gender": 0,
                            "age": 0,
                            "isFriend": true,
                            "isvip": true,
                            "isSvip": true,
                            "uin": 0
                        }
                    ]
                },
                "voteInfo": {
                    "total_count": 0,
                    "new_count": 0,
                    "new_nearby_count": 0,
                    "last_visit_time": 0,
                    "userInfos": [
                        {
                            "uid": "string",
                            "src": 0,
                            "latestTime": 0,
                            "count": 0,
                            "giftCount": 0,
                            "customId": 0,
                            "lastCharged": 0,
                            "bAvailableCnt": 0,
                            "bTodayVotedCnt": 0,
                            "nick": "string",
                            "gender": 0,
                            "age": 0,
                            "isFriend": true,
                            "isvip": true,
                            "isSvip": true,
                            "uin": 0
                        }
                    ]
                }
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        DELETE_FRIEND = "napcat_delete_friend"
        """删除好友
        
        Args:
            user_id (Optional[str|int]): 用户id(必需)
            temp_block (Optional[bool]): 拉黑(必需)
            temp_both_del (Optional[bool]): 双向删除(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
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
        """
        GET_USER_STATUS = "napcat_get_user_status"
        """获取(指定)用户状态
        
        Args:
            user_id (Optional[str|int]): 用户id(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "status": 0,
                "ext_status": 0
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_STATUS = "napcat_get_status"
        """获取状态
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "online": true,
                "good": true,
                "stat": {}
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_MINI_APP_ARK = "napcat_get_mini_app_ark"
        """获取小程序卡片
        
        Args:
            type (Optional[str]): 类型(如bili、weibo,必需)
            title (Optional[str]): 标题(必需)
            desc (Optional[str]): 描述(必需)
            picUrl (Optional[str]): 图片URL(必需)
            jumpUrl (Optional[str]): 跳转URL(必需)
            webUrl (Optional[str]): 网页URL
            rawArkData (Optional[bool]): 是否返回原始ark数据
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "appName": "string",
                "appView": "string",
                "ver": "string",
                "desc": "string",
                "prompt": "string",
                "metaData": {
                    "detail_1": {
                        "appid": "string",
                        "appType": 0,
                        "title": "string",
                        "desc": "string",
                        "icon": "string",
                        "preview": "string",
                        "url": "string",
                        "scene": 0,
                        "host": {
                            "uin": 0,
                            "nick": "string"
                        },
                        "shareTemplateId": "string",
                        "shareTemplateData": {},
                        "showLittleTail": "string",
                        "gamePoints": "string",
                        "gamePointsUrl": "string",
                        "shareOrigin": 0
                    }
                },
                "config": {
                    "type": "string",
                    "width": 0,
                    "height": 0,
                    "forward": 0,
                    "autoSize": 0,
                    "ctime": 0,
                    "token": "string"
                }
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SET_DIY_ONLINE_STATUS = "napcat_set_diy_online_status"
        """设置自定义在线状态
        
        Args:
            face_id (Optional[str|int]): 表情ID(必需)
            face_type (Optional[str|int]): 表情类型
            wording (Optional[str]):描述文本
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": "string",
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """

    class MESSAGE(Enum):
        """
        该分类是对信息相关的操作，只能由外部触发，napcat_plugin负责处理
        """

        SEND_PRIVATE_MSG = "napcat_send_private_msg"
        """发送私聊消息
        
        Args:
            user_id (Optional[str|int]): 用户id(必需)
            message (Optional[str]): 消息object(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "message_id": 0
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SEND_POKE = "napcat_send_poke"
        """发送戳一戳
        
        Args:
            group_id (Optional[str|int]): 群号
            user_id (Optional[str|int]): 对方QQ号(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        DELETE_MSG = "napcat_delete_msg"
        """撤回消息

        Args:
            message_id (Optional[str|int]): 消息id(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_GROUP_MSG_HISTORY = "napcat_get_group_msg_history"
        """获取群历史消息
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            message_seq (Optional[str|int]): 消息序号,0为最新
            count (Optional[int]): 获取数量
            reverseOrder (Optional[bool]): 是否倒序
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "messages": [
                    {
                        "self_id": 0,
                        "user_id": 0,
                        "time": 0,
                        "message_id": 0,
                        "message_seq": 0,
                        "real_id": 0,
                        "real_seq": "string",
                        "message_type": "string",
                        "sender": {
                            "user_id": 0,
                            "nickname": "string",
                            "sex": "male",
                            "age": 0,
                            "card": "string",
                            "role": "owner"
                        },
                        "raw_message": "string",
                        "font": 0,
                        "sub_type": "string",
                        "message": [
                            {
                                "type": "text",
                                "data": {
                                    "text": "string"
                                }
                            }
                        ],
                        "message_format": "string",
                        "post_type": "string",
                        "group_id": 0
                    }
                ]
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_MSG = "napcat_get_msg"
        """获取消息详情
        
        Args:
            message_id (Optional[str|int]): 消息id(必需)
            raw (Optional[dict]): 原始请求体

        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "self_id": 0,
                "user_id": 0,
                "time": 0,
                "message_id": 0,
                "message_seq": 0,
                "real_id": 0,
                "real_seq": "string",
                "message_type": "string",
                "sender": {
                    "user_id": 0,
                    "nickname": "string",
                    "sex": "male",
                    "age": 0,
                    "card": "string",
                    "role": "owner"
                },
                "raw_message": "string",
                "font": 0,
                "sub_type": "string",
                "message": [
                    {
                        "type": "text",
                        "data": {
                            "text": "string"
                        }
                    }
                ],
                "message_format": "string",
                "post_type": "string",
                "group_id": 0
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }        
        """
        GET_FORWARD_MSG = "napcat_get_forward_msg"
        """获取合并转发消息
        
        Args:
            message_id (Optional[str|int]): 消息id(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "messages": [
                    {
                        "self_id": 0,
                        "user_id": 0,
                        "time": 0,
                        "message_id": 0,
                        "message_seq": 0,
                        "real_id": 0,
                        "real_seq": "string",
                        "message_type": "string",
                        "sender": {
                            "user_id": 0,
                            "nickname": "string",
                            "sex": "male",
                            "age": 0,
                            "card": "string",
                            "role": "owner"
                        },
                        "raw_message": "string",
                        "font": 0,
                        "sub_type": "string",
                        "message": [
                            {
                                "type": "text",
                                "data": {
                                    "text": "string"
                                }
                            }
                        ],
                        "message_format": "string",
                        "post_type": "string",
                        "group_id": 0
                    }
                ]
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SET_MSG_EMOJI_LIKE = "napcat_set_msg_emoji_like"
        """贴表情
        
        Args:
            message_id (Optional[str|int]): 消息id(必需)
            emoji_id (Optional[int]): 表情id(必需)
            set (Optional[bool]): 是否贴(必需)
            raw (Optional[dict]): 原始请求体

        Returns:
            dict: {
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
        """
        GET_FRIEND_MSG_HISTORY = "napcat_get_friend_msg_history"
        """获取好友历史消息
        
        Args:
            user_id (Optional[str|int]): 用户id(必需)
            message_seq (Optional[str|int]): 消息序号,0为最新
            count (Optional[int]): 获取数量
            reverseOrder (Optional[bool]): 是否倒序
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "messages": [
                    {
                        "self_id": 0,
                        "user_id": 0,
                        "time": 0,
                        "message_id": 0,
                        "message_seq": 0,
                        "real_id": 0,
                        "real_seq": "string",
                        "message_type": "string",
                        "sender": {
                            "user_id": 0,
                            "nickname": "string",
                            "sex": "male",
                            "age": 0,
                            "card": "string",
                            "role": "owner"
                        },
                        "raw_message": "string",
                        "font": 0,
                        "sub_type": "string",
                        "message": [
                            {
                                "type": "text",
                                "data": {
                                    "text": "string"
                                }
                            }
                        ],
                        "message_format": "string",
                        "post_type": "string",
                        "group_id": 0
                    }
                ]
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        FETCH_EMOJI_LIKE = "napcat_fetch_emoji_like"
        """获取贴表情详情
        
        Args:
            message_id (Optional[str|int]): 消息id(必需)
            emojiId (Optional[str]): 表情id(必需)
            emojiType (Optional[str]): 表情类型(必需)
            count (Optional[int]): 返回数量
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "result": 0,
                "errMsg": "string",
                "emojiLikesList": [
                    {
                        "tinyId": "string",
                        "nickName": "string",
                        "headUrl": "string"
                    }
                ],
                "cookie": "string",
                "isLastPage": true,
                "isFirstPage": true
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SEND_FORWARD_MSG = "napcat_send_forward_msg"
        """发送合并转发消息
        
        Args:
            group_id (Optional[str|int]): 群号
            user_id (Optional[str|int]): 用户id
            messages (Optional[dict]): 一级合并转发消息节点(必需)
            news (Optional[dict]): 原转发消息之外的消息(必需)
            prompt (Optional[str]): 外显(必需)
            summary (Optional[str]): 底下文本(必需)
            source (Optional[str]): 内容(必需)
            raw (Optional[dict]): 原始请求体
            
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {},
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SEND_GROUP_AI_RECORD = "napcat_send_group_ai_record"
        """发送群AI语音
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            character (Optional[str]): 角色id(必需)
            text (Optional[str]): 文本(必需)
            raw (Optional[dict]): 原始请求体

        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "message_id": "string"
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """

    class GROUP(Enum):
        """
        该分类是对群聊相关的操作，只能由外部触发，napcat_plugin负责处理
        """

        GET_GROUP_INFO = "napcat_get_group_info"
        """获取群信息
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            raw (Optional[dict]): 原始请求体

        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "group_all_shut": 0,
                "group_remark": "string",
                "group_id": "string",
                "group_name": "string",
                "member_count": 0,
                "max_member_count": 0
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SET_GROUP_ADD_OPTION = "napcat_set_group_add_option"
        """设置群添加选项
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            add_type (Optional[str]): 群添加类型(必需)
            group_question (Optional[str]): 群添加问题
            group_answer (Optional[str]): 群添加答案
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SET_GROUP_KICK_MEMBERS = "napcat_set_group_kick_members"
        """批量踢出群成员
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            user_id (Optional[List[str|int]]): 用户id列表(必需)
            reject_add_request (Optional[bool]): 是否群拉黑
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SET_GROUP_REMARK = "napcat_set_group_remark"
        """设置群备注

        Args:
            group_id (Optional[str]): 群号(必需)
            remark (Optional[str]): 备注内容(必需)
            raw (Optional[dict]): 原始请求体

        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }     
        """
        SET_GROUP_KICK = "napcat_set_group_kick"
        """群踢人
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            user_id (Optional[str|int]): 用户id(必需)
            reject_add_request (Optional[bool]): 是否群拉黑
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_GROUP_SYSTEM_MSG = "napcat_get_group_system_msg"
        """获取群系统消息
        
        Args:
            count (Optional[int]): 获取数量(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "InvitedRequest": [
                    {
                        "request_id": 0,
                        "invitor_uin": 0,
                        "invitor_nick": "string",
                        "group_id": 0,
                        "message": "string",
                        "group_name": "string",
                        "checked": true,
                        "actor": 0,
                        "requester_nick": "string"
                    }
                ],
                "join_requests": [
                    {
                        "request_id": 0,
                        "invitor_uin": 0,
                        "invitor_nick": "string",
                        "group_id": 0,
                        "message": "string",
                        "group_name": "string",
                        "checked": true,
                        "actor": 0,
                        "requester_nick": "string"
                    }
                ]
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SET_GROUP_BAN = "napcat_set_group_ban"
        """群禁言
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            user_id (Optional[str|int]): 用户id(必需)
            duration (Optional[int]): 禁言时间：秒(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_ESSENCE_MSG_LIST = "napcat_get_essence_msg_list"
        """获取群精华消息
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": [
                {
                    "msg_seq": 0,
                    "msg_random": 0,
                    "sender_id": 0,
                    "sender_nick": "string",
                    "operator_id": 0,
                    "operator_nick": "string",
                    "message_id": 0,
                    "operator_time": 0,
                    "content": [
                        {
                            "type": "text",
                            "data": {
                                "text": "string"
                            }
                        }
                    ]
                }
            ],
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SET_GROUP_WHOLE_BAN = "napcat_set_group_whole_ban"
        """全体禁言
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            enable (Optional[bool]): 是否启用(必需)
            raw (Optional[dict]): 原始请求体

        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SET_GROUP_PORTRAINT = "napcat_set_group_portrait"
        """设置群头像
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            file (Optional[str]): 文件路径(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "result": 0,
                "errMsg": "success"
            },
            "message": "",
            "wording": "",
            "echo": null
        }
        """
        SET_GROUP_ADMIN = "napcat_set_group_admin"
        """设置群管理
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            user_id (Optional[str|int]): 用户id(必需)
            enable (Optional[bool]): 是否设为群管理(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SET_GROUP_CARD = "napcat_group_card"
        """设置群成员名片
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            user_id (Optional[str|int]): 用户id(必需)
            card (Optional[str]): 为空则为取消群名片
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SET_ESSENCE_MSG = "napcat_set_essence_msg"
        """设置群精华消息
        
        Args:
            message_id (Optional[str|int]): 消息id(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "errCode": 0,
                "errMsg": "success",
                "result": {
                    "wording": "",
                    "digestUin": "0",
                    "digestTime": 0,
                    "msg": {
                        "groupCode": "0",
                        "msgSeq": 0,
                        "msgRandom": 0,
                        "msgContent": [],
                        "textSize": "0",
                        "picSize": "0",
                        "videoSize": "0",
                        "senderUin": "0",
                        "senderTime": 0,
                        "addDigestUin": "0",
                        "addDigestTime": 0,
                        "startTime": 0,
                        "latestMsgSeq": 0,
                        "opType": 0
                    },
                    "errorCode": 0
                }
            },
            "message": "",
            "wording": "",
            "echo": null
        }
        """
        SET_GROUP_NAME = "napcat_set_group_name"
        """设置群名
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            group_name (Optional[str]): 群名(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        DELETE_ESSENCE_MSG = "napcat_delete_essence_msg"
        """删除群精华消息
        
        Args:
            message_id (Optional[str|int]): 消息id(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict:{
            "status": "ok",
            "retcode": 0,
            "data": {
                "errCode": 0,
                "errMsg": "success",
                "result": {
                    "wording": "",
                    "digestUin": "0",
                    "digestTime": 0,
                    "msg": {
                        "groupCode": "0",
                        "msgSeq": 0,
                        "msgRandom": 0,
                        "msgContent": [],
                        "textSize": "0",
                        "picSize": "0",
                        "videoSize": "0",
                        "senderUin": "0",
                        "senderTime": 0,
                        "addDigestUin": "0",
                        "addDigestTime": 0,
                        "startTime": 0,
                        "latestMsgSeq": 0,
                        "opType": 0
                    },
                    "errorCode": 0
                }
            },
            "message": "",
            "wording": "",
            "echo": null
        }
        """
        SET_GROUP_LEAVE = "napcat_set_group_leave"
        """退群
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SEND_GROUP_NOTICE = "napcat_group_notice"
        """发送群公告
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            content (Optional[str]): 公告内容(必需)
            image (Optional[str]): 图片地址
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SET_GROUP_SPECIAL_TITLE = "napcat_set_group_special_title"
        """设置群头衔
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            user_id (Optional[str|int]): 用户id(必需)
            special_title (Optional[str]): 为空则取消头衔
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_GROUP_NOTICE = "napcat_get_group_notice"
        """获取群公告
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            raw (Optional[dict]): 原始请求体
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": [
                {
                    "notice_id": "63491e2f000000004f4d1e677d2b0200",
                    "sender_id": 123,
                    "publish_time": 1730039119,
                    "message": {
                        "text": "这是一条神奇的群公告",
                        "image": [
                            {
                                "id": "aJJBbZ6BqyLiaC1kmpvIWGBBkJerEfpRBHX5Brxbaurs",
                                "height": "400",
                                "width": "400"
                            }
                        ]
                    }
                }
            ],
            "message": "",
            "wording": "",
            "echo": null
        }
        """
        SET_GROUP_ADD_REQUEST = "napcat_set_group_add_request"
        """处理加群请求
        
        Args:
            flag (Optional[str]): 请求id(必需)
            approve (Optional[bool]): 是否同意(必需)
            reason (Optional[str]): 拒绝理由
            raw (Optional[dict]): 原始请求体

        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": null,
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_GROUP_LIST = "napcat_get_group_list"
        """获取群列表
        
        Args:
            no_cache (Optional[bool]): 是否不缓存
            raw (Optional[dict]): 原始请求体  
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": [
                {
                    "group_all_shut": 0,
                    "group_remark": "string",
                    "group_id": "string",
                    "group_name": "string",
                    "member_count": 0,
                    "max_member_count": 0
                }
            ],
            "message": "string",
            "wording": "string",
            "echo": "string"
        }   
        """
        DELETE_GROUP_NOTICE = "napcat_del_group_notice"
        """删除群公告
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            notice_id (Optional[str]): 公告id(必需)
            raw (Optional[dict]): 原始请求体  
        
        Returns:
            dict: {
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
        """
        GET_GROUP_MEMBER_INFO = "napcat_get_group_member_info"
        """获取群成员信息
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            user_id (Optional[str|int]): 用户id(必需)
            no_cache (Optional[bool]): 是否不缓存
            raw (Optional[dict]): 原始请求体 
    
        Returns:
            dict:{
            "status": "ok",
            "retcode": 0,
            "data": {
                "group_id": 0,
                "user_id": 0,
                "nickname": "string",
                "card": "string",
                "sex": "string",
                "age": 0,
                "join_time": 0,
                "last_sent_time": 0,
                "level": 0,
                "qq_level": 0,
                "role": "string",
                "title": "string",
                "area": "string",
                "unfriendly": true,
                "title_expire_time": 0,
                "card_changeable": true,
                "shut_up_timestamp": 0,
                "is_robot": true,
                "qage": "string"
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_GROUP_MEMBER_LIST = "napcat_get_group_member_list"
        """获取群成员列表
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            no_cache (Optional[bool]): 是否不缓存
            raw (Optional[dict]): 原始请求体 
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": [
                {
                    "group_id": 0,
                    "user_id": 0,
                    "nickname": "string",
                    "card": "string",
                    "sex": "string",
                    "age": 0,
                    "join_time": 0,
                    "last_sent_time": 0,
                    "level": 0,
                    "qq_level": 0,
                    "role": "string",
                    "title": "string",
                    "area": "string",
                    "unfriendly": true,
                    "title_expire_time": 0,
                    "card_changeable": true,
                    "shut_up_timestamp": 0,
                    "is_robot": true,
                    "qage": "string"
                }
            ],
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_GROUP_HONOR_INFO = "napcat_get_group_honor_info"
        """获取群荣誉
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            type (Optional[str]): 看详情
            raw (Optional[dict]): 原始请求体 
        
        Returns:
            dict:{
            "status": "ok",
            "retcode": 0,
            "data": {
                "group_id": "string",
                "current_talkative": {
                    "user_id": 0,
                    "nickname": "string",
                    "avatar": 0,
                    "description": "string"
                },
                "talkative_list": [
                    {
                        "user_id": 0,
                        "nickname": "string",
                        "avatar": 0,
                        "description": "string"
                    }
                ],
                "performer_list": [
                    {
                        "user_id": 0,
                        "nickname": "string",
                        "avatar": 0,
                        "description": "string"
                    }
                ],
                "legend_list": [
                    {
                        "user_id": 0,
                        "nickname": "string",
                        "avatar": 0,
                        "description": "string"
                    }
                ],
                "emotion_list": [
                    {
                        "user_id": 0,
                        "nickname": "string",
                        "avatar": 0,
                        "description": "string"
                    }
                ],
                "strong_newbie_list": [
                    {
                        "user_id": 0,
                        "nickname": "string",
                        "avatar": 0,
                        "description": "string"
                    }
                ]
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_GROUP_INFO_EX = "napcat_get_group_info_ex"
        """获取群信息ex
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            raw (Optional[dict]): 原始请求体 
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "groupCode": "790514019",
                "resultCode": 0,
                "extInfo": {
                    "groupInfoExtSeq": 1,
                    "reserve": 0,
                    "luckyWordId": "0",
                    "lightCharNum": 0,
                    "luckyWord": "",
                    "starId": 0,
                    "essentialMsgSwitch": 0,
                    "todoSeq": 0,
                    "blacklistExpireTime": 0,
                    "isLimitGroupRtc": 0,
                    "companyId": 0,
                    "hasGroupCustomPortrait": 1,
                    "bindGuildId": "0",
                    "groupOwnerId": {
                        "memberUin": "1129317309",
                        "memberUid": "u_4_QA-QaFryh-Ocgsv4_8EQ",
                        "memberQid": ""
                    },
                    "essentialMsgPrivilege": 0,
                    "msgEventSeq": "0",
                    "inviteRobotSwitch": 0,
                    "gangUpId": "0",
                    "qqMusicMedalSwitch": 0,
                    "showPlayTogetherSwitch": 0,
                    "groupFlagPro1": "0",
                    "groupBindGuildIds": {
                        "guildIds": []
                    },
                    "viewedMsgDisappearTime": "0",
                    "groupExtFlameData": {
                        "switchState": 0,
                        "state": 0,
                        "dayNums": [],
                        "version": 0,
                        "updateTime": "0",
                        "isDisplayDayNum": false
                    },
                    "groupBindGuildSwitch": 0,
                    "groupAioBindGuildId": "0",
                    "groupExcludeGuildIds": {
                        "guildIds": []
                    },
                    "fullGroupExpansionSwitch": 0,
                    "fullGroupExpansionSeq": "0",
                    "inviteRobotMemberSwitch": 0,
                    "inviteRobotMemberExamine": 0,
                    "groupSquareSwitch": 0
                }
            },
            "message": "",
            "wording": "",
            "echo": null
        }      
        """
        GET_GROUP_AT_ALL_REMAIN = "napcat_get_group_at_all_remain"
        """获取群 @全体成员 剩余次数
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            raw (Optional[dict]): 原始请求体 
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "can_at_all": true,
                "remain_at_all_count_for_group": 0,
                "remain_at_all_count_for_uin": 0
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_GROUP_SHUT_LIST = "napcat_get_group_shut_list"
        """获取群禁言列表
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            raw (Optional[dict]): 原始请求体 
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": [
                {
                    "uid": "string",
                    "qid": "string",
                    "uin": "string",
                    "nick": "string",
                    "remark": "string",
                    "cardType": 0,
                    "cardName": "string",
                    "role": 0,
                    "avatarPath": "string",
                    "shutUpTime": 0,
                    "isDelete": true,
                    "isSpecialConcerned": true,
                    "isSpecialShield": true,
                    "isRobot": true,
                    "groupHonor": {},
                    "memberRealLevel": 0,
                    "memberLevel": 0,
                    "globalGroupLevel": 0,
                    "globalGroupPoint": 0,
                    "memberTitleId": 0,
                    "memberSpecialTitle": "string",
                    "specialTitleExpireTime": "string",
                    "userShowFlag": 0,
                    "userShowFlagNew": 0,
                    "richFlag": 0,
                    "mssVipType": 0,
                    "bigClubLevel": 0,
                    "bigClubFlag": 0,
                    "autoRemark": "string",
                    "creditLevel": 0,
                    "joinTime": 0,
                    "lastSpeakTime": 0,
                    "memberFlag": 0,
                    "memberFlagExt": 0,
                    "memberMobileFlag": 0,
                    "memberFlagExt2": 0,
                    "isSpecialShielded": true,
                    "cardNameId": 0
                }
            ],
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        GET_GROUP_IGNORED_NOTIFIES = "napcat_get_group_ignored_notifies"
        """获取群过滤系统消息
        
        Returns:
            dict: {
            "status": "ok",
            "retcode": 0,
            "data": {
                "InvitedRequest": [
                    {
                        "request_id": 0,
                        "invitor_uin": 0,
                        "invitor_nick": "string",
                        "group_id": 0,
                        "message": "string",
                        "group_name": "string",
                        "checked": true,
                        "actor": 0,
                        "requester_nick": "string"
                    }
                ],
                "join_requests": [
                    {
                        "request_id": 0,
                        "invitor_uin": 0,
                        "invitor_nick": "string",
                        "group_id": 0,
                        "message": "string",
                        "group_name": "string",
                        "checked": true,
                        "actor": 0,
                        "requester_nick": "string"
                    }
                ]
            },
            "message": "string",
            "wording": "string",
            "echo": "string"
        }
        """
        SET_GROUP_SIGN = "napcat_set_group_sign"
        """群打卡
        
        Args:
            group_id (Optional[str|int]): 群号(必需)
            raw (Optional[dict]): 原始请求体 

        Returns:
            dict: {}
        """

    class FILE(Enum): ...

    class PERSONAL(Enum):
        SET_INPUT_STATUS = "napcat_set_input_status"
        """
        设置输入状态

        Args:
            user_id (Optional[str|int]): 用户id(必需)
            event_type (Optional[int]): 输入状态id(必需)
            raw (Optional[dict]): 原始请求体 

        Returns:
            dict: {
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
        """
