from typing import List, Dict, Any, Optional, Callable
from langchain.schema import Document
from typing import Protocol

class DocumentProcessor(Protocol):
    """文档处理器接口"""
    
    def load_and_index_files(
        self,
        processing_callback: Callable[[List[Document]], None]
    ) -> None:
        """加载并索引COS文档
        
        Args:
            processing_callback: 文档处理回调函数，必须提供
                函数签名应为 (List[Document]) -> None
                示例用法：
                ```python
                def custom_processor(splits):
                    # 数据增强
                    enhanced = enhance_documents(splits)
                    # 存储处理
                    vector_store.add_documents(enhanced)
                    # 额外处理（如日志、监控等）
                    log_processing_metrics(len(splits))
                ```
        
        Raises:
            ValueError: 当未提供回调函数或回调函数不可调用时抛出
        """
        ...
    
    def clear_all(self) -> None:
        """清空所有文档和向量数据
        
        说明:
            删除所有存储的文档和相关的向量数据，恢复到初始状态。
        """
        ...
    
    def add_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """添加文档
        
        参数:
            content: 文档的内容（字符串格式）。
            metadata: 文档的元数据（如标题、分类、标签等），格式为字典。
        
        返回值:
            添加的文档的唯一标识符（文档ID）。
        
        说明:
            将文档内容和元数据添加到存储中，并生成文档ID。
        """
        ...
    
    def remove_document(self, doc_id: str) -> None:
        """删除文档
        
        参数:
            doc_id: 要删除的文档的唯一标识符（文档ID）。
        
        说明:
            根据文档ID删除指定的文档及其相关数据。
        """
        ...
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取文档
        
        参数:
            doc_id: 要获取的文档的唯一标识符（文档ID）。
        
        返回值:
            文档的信息（包括内容和元数据），格式为字典。
            如果文档不存在，则返回 None。
        
        说明:
            根据文档ID获取文档的详细信息。
        """
        ...
    
    def search_documents(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Document]:
        """搜索文档
        
        参数:
            query: 查询文本，用于搜索相关文档。
            filter_dict: 可选的过滤条件，格式为字典（如按元数据字段过滤）。
            top_k: 返回的相关文档数量，默认为 5。
        
        返回值:
            相关文档的列表，每个文档为 langchain.schema.Document 对象。
        
        说明:
            根据查询文本和过滤条件搜索相关文档，返回最匹配的 top_k 个结果。
        """
        ...