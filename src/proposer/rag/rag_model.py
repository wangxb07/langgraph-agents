from typing import Any, List, Optional, Mapping
from langchain.callbacks.manager import PLACEHOLDER_FOR_SECRET_ID
from langchain.llms.base import LLM
from langchain.schema import HumanMessage
from pydantic.v1 import Field, PrivateAttr

from models.base import BaseQwenModel


class ConcreteQwenModel(BaseQwenModel):
    """具体的Qwen模型实现，用于RAG场景"""
    
    def __init__(self, model: str = "qwen-plus", api_version: str = "v1", agent_name: str = "rag", temperature: float = 0.7):
        """初始化具体的Qwen模型
        
        Args:
            model: 模型名称，默认为qwen-plus
            api_version: API版本，默认为v1
            agent_name: agent名称，用于LangSmith tracking
            temperature: 模型采样温度，控制输出的随机性
        """
        super().__init__(model=model, api_version=api_version, agent_name=agent_name)
        self.temperature = temperature
    
    async def ainvoke(self, messages: List[Any], step_id: str) -> str:
        """实现BaseQwenModel的抽象方法
        
        Args:
            messages: 消息列表
            step_id: 步骤ID
            
        Returns:
            str: 模型生成的文本
        """
        return await self._base_ainvoke(messages, step_id, temperature=self.temperature)


class RAGQwenModel(LLM):
    """用于RAG的Qwen模型，基于LangChain的LLM基类"""
    
    model: str = Field(default="qwen-plus")
    api_version: str = Field(default="v1")
    agent_name: str = Field(default="rag")
    temperature: float = Field(default=0.7)
    _base_model: ConcreteQwenModel = PrivateAttr()
    
    def __init__(self, **kwargs):
        """初始化RAG Qwen模型
        
        Args:
            model: 模型名称，默认为qwen-plus
            api_version: API版本，默认为v1
            agent_name: agent名称，用于LangSmith tracking
            temperature: 模型采样温度，控制输出的随机性
        """
        super().__init__(**kwargs)
        # 初始化基础客户端
        self._base_model = ConcreteQwenModel(
            model=self.model,
            api_version=self.api_version,
            agent_name=self.agent_name,
            temperature=self.temperature
        )

    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "qwen"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[PLACEHOLDER_FOR_SECRET_ID] = None,
        **kwargs: Any,
    ) -> str:
        """实现LangChain LLM基类的_call方法
        
        Args:
            prompt: 输入提示
            stop: 停止词列表
            run_manager: 回调管理器
            **kwargs: 其他参数
            
        Returns:
            str: 模型生成的文本
        """
        import asyncio
        
        # 将prompt转换为消息格式
        messages = [HumanMessage(content=prompt)]
        
        # 使用asyncio运行异步调用
        response = asyncio.run(self._base_model.ainvoke(messages, step_id="rag_call"))
        return response
