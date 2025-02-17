import pytest
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models.tongyi import ChatTongyi

# 确保环境变量已设置
assert "DASHSCOPE_API_KEY" in os.environ, "DASHSCOPE_API_KEY 环境变量未设置"

@pytest.fixture(params=["qwen-max", "deepseek-v3", "deepseek-r1"])
def chat_model(request):
    """创建一个ChatTongyi模型实例用于测试"""
    model = ChatTongyi(model_name=request.param)
    yield model

@pytest.mark.asyncio
async def test_model_initialization(chat_model):
    """测试模型初始化"""
    assert chat_model.model_name in ["qwen-max", "deepseek-v3", "deepseek-r1"]

@pytest.mark.asyncio
async def test_basic_chat(chat_model):
    """测试基本的对话功能"""
    messages = [HumanMessage(content="Hello! What's 1+1?")]
    response = await chat_model.agenerate([messages])
    assert response.generations[0][0].text is not None

@pytest.mark.asyncio
async def test_chat_with_system_message(chat_model):
    """测试带有系统消息的对话"""
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="What's your purpose?")
    ]
    response = await chat_model.agenerate([messages])
    assert response.generations[0][0].text is not None

@pytest.mark.asyncio
async def test_chat_with_history(chat_model):
    """测试带有历史记录的对话"""
    messages = [
        HumanMessage(content="Hi, my name is Alice."),
        AIMessage(content="Hello Alice! Nice to meet you."),
        HumanMessage(content="What's my name?")
    ]
    response = await chat_model.agenerate([messages])
    assert "Alice" in response.generations[0][0].text
