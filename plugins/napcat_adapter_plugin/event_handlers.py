from typing import List, Tuple, Optional

from src.plugin_system import BaseEventHandler
from src.plugin_system.base.base_event import HandlerResult
from src.plugin_system.core.event_manager import event_manager

from .src.send_handler import send_handler
from .event_types import *

from src.common.logger import get_logger
logger = get_logger("napcat_adapter")


class SetProfileHandler(BaseEventHandler):
    handler_name: str = "napcat_set_qq_profile_handler"
    handler_description: str = "设置账号信息"
    weight: int = 100
    intercept_message: bool = False
    init_subscribe = [NapcatEvent.ACCOUNT.SET_PROFILE]

    async def execute(self,params:dict):
        raw = params.get("raw",{})
        nickname = params.get("nickname","")
        personal_note = params.get("personal_note","")
        sex = params.get("sex","")

        if params.get("raw",""):
            nickname = raw.get("nickname","")
            personal_note = raw.get("personal_note","")
            sex = raw.get("sex","")
        
        if not nickname:
            logger.error("事件 napcat_set_qq_profile 缺少必要参数: nickname ")
            return HandlerResult(False,False,{"status":"error"})

        payload = {
            "nickname": nickname,
            "personal_note": personal_note,
            "sex": sex
            }
        response = await send_handler.send_message_to_napcat(action="set_qq_profile",params=payload)
        if response.get("status","") == "ok":
            if response.get("data","").get("result","") == 0:
                return HandlerResult(True,True,response)
            else:
                logger.error(f"事件 napcat_set_qq_profile 请求失败！err={response.get("data","").get("errMsg","")}")
                return HandlerResult(False,False,response)
        else:
            logger.error("事件 napcat_set_qq_profile 请求失败！")
            return HandlerResult(False,False,{"status":"error"})
'''
class SetProfileHandler(BaseEventHandler):
    handler_name: str = "napcat_set_qq_profile_handler"
    handler_description: str = "设置账号信息"
    weight: int = 100
    intercept_message: bool = False
    init_subscribe = [NapcatEvent.ACCOUNT.SET_PROFILE]

    async def execute(
            self,
            nickname: Optional[str] = "", 
            personal_note: Optional[str] = "",
            sex: Optional[list["1","2","3"]] = "",
            raw: dict = {}
            ):
        if raw:
            nickname = raw.get("nickname","")
            personal_note = raw.get("personal_note","")
            sex = raw.get("sex","")
        
        if not nickname:
            logger.error("事件 napcat_set_qq_profile 缺少必要参数: nickname ")
            return HandlerResult(False,False,"缺少必要参数: nickname")

        payload = {
            "nickname": nickname,
            "personal_note": personal_note,
            "sex": sex
            }
        response = await send_handler.send_message_to_napcat(action="set_qq_profile",params=payload)
        if response.get("status","") == "ok":
            if response.get("data","").get("result","") == 0:
                return HandlerResult(True,True,True)
            else:
                logger.error(f"事件 napcat_set_qq_profile 请求失败！err={response.get("data","").get("errMsg","")}")
                return HandlerResult(False,False,False)
        else:
            logger.error("事件 napcat_set_qq_profile 请求失败！")
            return HandlerResult(False,False,False)
'''   

        
