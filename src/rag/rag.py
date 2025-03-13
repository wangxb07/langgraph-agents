import os
import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma

from .embeddings import DashScopeEmbeddings
from .rag_model import RAGQwenModel
from .document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = """使用以下上下文来回答问题。如果你不知道答案，就说你不知道，不要试图编造答案。

上下文：
{context}

问题：{question}

答案："""

class RAGTool:
    """RAG工具类
    
    用于从文档库中检索相关内容，增强提案生成。
    """
    
    def __init__(
        self,
        document_processor: DocumentProcessor,  
        embedding_model: str = "text-embedding-v2",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    ):
        """初始化RAG工具
        
        Args:
            document_processor: 文档处理器实现
            embedding_model: DashScope Embedding模型名称
            prompt_template: 提示模板
        """
        self.document_processor = document_processor
        self.vector_store_dir = os.path.join(os.getcwd(), "knowledge", "vector_store")
        
        logger.info(f"初始化RAG工具，向量存储目录：{self.vector_store_dir}")
        
        # 确保目录存在
        os.makedirs(self.vector_store_dir, exist_ok=True)
        
        self.embeddings = DashScopeEmbeddings(model=embedding_model)
        
        # 初始化向量数据库
        self.vectorstore = self._init_vectorstore()
        
        # 加载并索引文档，使用向量存储回调
        def vector_store_callback(splits: List[Document]):
            self.vectorstore.add_documents(splits)
            
        self.document_processor.load_and_index_files(
            processing_callback=vector_store_callback
        )
        
        # 初始化检索链
        self.qa_chain = self._init_qa_chain(prompt_template)
        
    def _init_vectorstore(self):
        """初始化向量数据库"""
        return Chroma(
            persist_directory=self.vector_store_dir,
            embedding_function=self.embeddings
        )

    def _init_qa_chain(self, prompt_template: str) -> RetrievalQA:
        """初始化问答链
        
        Args:
            prompt_template: 提示模板
            
        Returns:
            RetrievalQA 链
        """
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        llm = RAGQwenModel()
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
    def query(
        self,
        question: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """查询知识库
        
        Args:
            question: 问题
            category: 按分类筛选
            tags: 按标签筛选
            
        Returns:
            包含答案和来源文档的字典
        """
        result = self.qa_chain.invoke({"query": question})
        
        source_docs = []
        for doc in result.get("source_documents", []):
            metadata = doc.metadata
            source_docs.append({
                "content": doc.page_content,
                "title": metadata.get("title"),
                "category": metadata.get("category"),
                "source": metadata.get("source")
            })
            
        return {
            "answer": result["result"],
            "sources": source_docs
        }
        
    def retrieve(
        self,
        query: str,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """检索相关文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            相关文档列表，每个文档包含内容和元数据
        """
        logger.info(f"开始检索，查询：{query[:100]}...")
        
        docs = self.vectorstore.similarity_search(query, k=k)
        logger.info(f"向量检索返回 {len(docs)} 个结果")
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
        return results