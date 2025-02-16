import pytest
import os
import logging
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
from proposer.llm.base import BaseQwenModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # 强制重新配置日志
)

# 设置更详细的日志级别
logging.getLogger('proposer.llm.base').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# 设置测试环境变量
os.environ["DASHSCOPE_API_KEY"] = "sk-PLACEHOLDER_FOR_SECRET_ID"

@pytest.mark.asyncio
async def test_basic_chat():
    """测试最基本的对话功能"""
    try:
        # 1. 测试模型初始化
        logger.info("Initializing model...")
        model = BaseQwenModel(model="qwen-max")
        assert isinstance(model, BaseChatModel)
        
        # 2. 测试基本对话
        logger.info("Testing basic chat...")
        messages = [HumanMessage(content="Hello! What's 1+1?")]
        
        logger.info("Sending message to model...")
        # 使用agenerate而不是ainvoke
        response = await model.agenerate([messages])
        
        # 3. 验证响应
        logger.info(f"Response received: {response.generations[0][0].text}")
        assert response.generations[0][0].text is not None
        assert isinstance(response.generations[0][0].text, str)
        assert len(response.generations[0][0].text) > 0
        
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        raise
