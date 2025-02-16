from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import Dict, List, Any, Optional, Iterator, ClassVar
import os
from openai import AsyncOpenAI
import logging
from pydantic import PrivateAttr
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseQwenModel(BaseChatModel):
    """基础的Qwen模型类，提供通用的模型调用功能"""
    
    provider_map: ClassVar[Dict[str, str]] = {
        "qwen": "qwen-max",
        "qwen-max": "qwen-max",
        "qwen-plus": "qwen-plus",
        "qwen-turbo": "qwen-turbo",
    }
    
    # 使用PrivateAttr来存储实例属性
    _model: str = PrivateAttr(default="qwen-max")
    _client: Optional[AsyncOpenAI] = PrivateAttr(default=None)
    
    def __init__(self, model: str = "qwen-max"):
        """初始化Qwen模型
        
        Args:
            model: 模型名称，默认为qwen-max
        """
        super().__init__()
        
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is not set")
            
        logger.info(f"Initializing BaseQwenModel with model: {model}")
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    @property
    def model(self) -> str:
        """获取模型名称"""
        return self._model
    
    @property
    def client(self) -> AsyncOpenAI:
        """获取API客户端"""
        if self._client is None:
            raise ValueError("Client is not initialized")
        return self._client
        
    async def __aenter__(self):
        logger.info("Entering context manager")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.info("Exiting context manager")
        if self._client:
            await self._client.close()
    
    def _convert_to_openai_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """转换消息格式为OpenAI兼容格式"""
        role_mapping = {
            SystemMessage: "system",
            HumanMessage: "user",
            AIMessage: "assistant"
        }
        converted = [
            {
                "role": role_mapping.get(type(msg), "assistant"),
                "content": msg.content
            }
            for msg in messages
        ]
        logger.debug(f"Converted messages: {converted}")
        return converted
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步生成回复
        
        Args:
            messages: 消息列表的列表
            stop: 停止词列表
            run_manager: 运行管理器
            **kwargs: 其他参数
            
        Returns:
            ChatResult: 聊天结果
        """
        try:
            logger.info("Starting async message generation")
            # 确保messages是一个列表
            if not isinstance(messages, list):
                raise ValueError("Messages must be a list")
                
            # 获取第一组对话消息
            chat_messages = messages[0] if isinstance(messages[0], list) else messages
            openai_messages = self._convert_to_openai_messages(chat_messages)
            
            # 从kwargs中获取temperature，默认为1.0
            temperature = kwargs.get("temperature", 1.0)
            logger.info(f"Using temperature: {temperature}")
            
            logger.info("Calling OpenAI API...")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                PLACEHOLDER_FOR_SECRET_ID,
                stop=stop
            )
            
            if not response.choices:
                raise ValueError("Empty response from model")
            
            content = response.choices[0].message.content.strip()
            logger.info(f"Received response: {content[:100]}...")
            
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            chat_result = ChatResult(generations=[generation])  
            
            logger.info("Successfully generated response")
            return chat_result
            
        except Exception as e:
            logger.error(f"Model generation failed: {str(e)}")
            raise
            
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """同步生成回复（实际上是包装异步方法）
        
        Args:
            messages: 消息列表
            stop: 停止词列表
            run_manager: 运行管理器
            **kwargs: 其他参数
            
        Returns:
            ChatResult: 聊天结果
        """
        return asyncio.run(self._agenerate([messages], stop=stop, run_manager=run_manager, **kwargs))
            
    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "qwen"
