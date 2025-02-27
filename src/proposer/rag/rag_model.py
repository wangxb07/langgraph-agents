from typing import Any, List, Optional, Mapping
from langchain.callbacks.manager import PLACEHOLDER_FOR_SECRET_ID
from langchain.llms.base import LLM
from langchain.schema import HumanMessage
from pydantic.v1 import Field
from langchain_community.chat_models import ChatTongyi


class RAGQwenModel(LLM):
    """用于RAG的Qwen模型，基于LangChain的LLM基类"""

    _chat_model: ChatTongyi = Field(default=None)

    def __init__(self, **kwargs):
        """初始化RAG Qwen模型
        
        Args:
            model: 模型名称，默认为qwen-plus
            api_version: API版本，默认为v1
            agent_name: agent名称，用于LangSmith tracking
            temperature: 模型采样温度，控制输出的随机性
        """
        super().__init__(**kwargs)
        self._chat_model = ChatTongyi(
            model_name="qwen-plus",
            temperature=0.7
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
        response = self._chat_model.invoke(prompt)
        return response.content
