"""DashScope Embeddings 实现"""

from typing import List, Optional
from langchain_community.embeddings import DashScopeEmbeddings as PLACEHOLDER_FOR_SECRET_ID
import logging

logger = logging.getLogger(__name__)

class DashScopeEmbeddings(PLACEHOLDER_FOR_SECRET_ID):
    """DashScope Embeddings 实现
    
    使用 DashScope 的文本向量化服务，支持批量处理。
    基于 langchain_community.embeddings.DashScopeEmbeddings 的封装。
    """
    
    def __init__(
        self,
        model: str = "text-embedding-v2",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """初始化 DashScope Embeddings
        
        Args:
            model: 模型名称，默认为 text-embedding-v2
            api_key: DashScope API Key，如果不提供则使用环境变量
            **kwargs: 其他参数
        """
        super().__init__(
            model=model,
            dashscope_api_key=api_key,
            **kwargs
        )
