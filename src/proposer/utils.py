from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_openai import ChatOpenAI
import os
from typing import Union, Optional

def init_custom_chat_model(model_name: str, api_version: str = "v1") -> Union[ChatTongyi, ChatOpenAI]:
    """初始化并返回一个聊天模型实例。
    
    支持通义千问系列模型和通过DashScope兼容OpenAI API的其他模型。
    
    Args:
        model_name (str): 模型名称，例如 "qwen-max", "deepseek-r1" 等
        api_version (str, optional): API版本，默认为 "v1"
        
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
