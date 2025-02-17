from langchain_community.chat_models.tongyi import ChatTongyi

def init_custom_chat_model(model_name: str) -> ChatTongyi:
    """初始化并返回一个 ChatTongyi 模型实例。
    
    Args:
        model_name (str): 模型名称，例如 "qwen-max"
        
    Returns:
        ChatTongyi: 初始化的 ChatTongyi 模型实例
    """
    return ChatTongyi(model_name=model_name)
