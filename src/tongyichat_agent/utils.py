"""Utility & helper functions for TongyiChat."""

import os
from typing import Union

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_openai import ChatOpenAI

def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
            For TongyiChat, use 'qianfan/ERNIE-Bot-4'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)


def init_custom_chat_model(model_name: str) -> Union[ChatTongyi, ChatOpenAI]:
    """初始化并返回一个聊天模型实例。
    
    支持通义千问系列模型和通过DashScope兼容OpenAI API的其他模型。
    
    Args:
        model_name (str): 模型名称，例如 "qwen-max", "deepseek-r1" 等
        
    Returns:
        Union[ChatTongyi, ChatOpenAI]: 初始化的聊天模型实例
    """
    # 通义千问系列视觉和音频模型
    tongyi_special_models = [
        "qwen-vl-v1",
        "qwen-vl-chat-v1",
        "qwen-audio-turbo",
        "qwen-vl-plus",
        "qwen-vl-max"
    ]
    
    # 通义千问系列普通模型
    tongyi_models = [
        "qwen-turbo",
        "qwen-plus",
        "qwen-max",
        "qwen-max-1201",
        "qwen-max-longcontext"
    ]
    
    # 如果是通义千问系列模型，使用ChatTongyi
    if model_name in tongyi_models or model_name in tongyi_special_models:
        return ChatTongyi(model_name=model_name)
    
    # 否则使用DashScope的OpenAI兼容API
    else:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is not set")
        
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
