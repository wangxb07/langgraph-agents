import os
import logging
import shutil
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.text_splitter import PLACEHOLDER_FOR_SECRET_ID
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
        
        # 确保目录存在并且有正确的权限
        os.makedirs(self.vector_store_dir, mode=0o777, exist_ok=True)
        os.chmod(self.vector_store_dir, 0o777)
        
        self.embeddings = DashScopeEmbeddings(model=embedding_model)
        self.text_splitter = PLACEHOLDER_FOR_SECRET_ID(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n## ", "\n# ", "\n\n", "\n", ". ", "? ", "! ", "；", "。", " ", ""]
        )
        
        # 初始化或加载向量数据库
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
        persist_directory = self.vector_store_dir
        
        # 删除旧的向量数据库
        if os.path.exists(persist_directory):
            logger.info(f"删除旧的向量数据库：{persist_directory}")
            shutil.rmtree(persist_directory)
        
        # 初始化向量数据库
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        return vectorstore

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
        result = self.qa_chain({"query": question})
        
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
        k: int = 3,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """检索相关文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            category: 按分类筛选
            tags: 按标签筛选
            
        Returns:
            相关文档列表，每个文档包含内容和元数据
        """
        logger.info(f"开始检索，查询：{query[:100]}...")
        
        docs = self.vectorstore.similarity_search(query, k=k*2)
        logger.info(f"向量检索返回 {len(docs)} 个结果")
        
        filtered_docs = []
        seen_doc_ids = set()
        
        for doc in docs:
            doc_id = doc.metadata["doc_id"]
            
            if doc_id in seen_doc_ids:
                continue
                
            if category and doc.metadata["category"] != category:
                continue
                
            if tags:
                doc_info = self.document_processor.get_document(doc_id)
                if not doc_info or not any(tag in doc_info["tags"] for tag in tags):
                    continue
                    
            filtered_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            seen_doc_ids.add(doc_id)
            
            if len(filtered_docs) >= k:
                break
                
        logger.info(f"筛选后返回 {len(filtered_docs)} 个结果")
        return filtered_docs
        
    def get_vector_store_info(self) -> Dict[str, Any]:
        """获取向量数据库信息"""
        collection = self.vectorstore._collection
        results = collection.get(where={}, include=['metadatas'])
        
        if not results or 'metadatas' not in results:
            return {"doc_count": 0, "chunk_count": 0, "documents": []}
            
        doc_chunks = {}
        for metadata in results['metadatas']:
            doc_id = metadata.get('doc_id')
            if doc_id:
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = {'title': metadata.get('title', 'Unknown'), 'chunks': []}
                doc_chunks[doc_id]['chunks'].append(metadata.get('chunk_id', -1))
                
        documents = [
            {'doc_id': doc_id, 'title': info['title'], 'chunk_count': len(info['chunks'])}
            for doc_id, info in doc_chunks.items()
        ]
        
        return {
            "doc_count": len(doc_chunks),
            "chunk_count": len(results['metadatas']),
            "documents": documents
        }
        
    def print_vector_store_status(self):
        """打印向量数据库状态"""
        info = self.get_vector_store_info()
        logger.info("=== 向量数据库状态 ===")
        logger.info(f"文档总数: {info['doc_count']}")
        logger.info(f"文档块总数: {info['chunk_count']}")
        logger.info("\n文档列表:")
        for doc in info['documents']:
            logger.info(f"- {doc['title']} (ID: {doc['doc_id']}, 块数: {doc['chunk_count']})")
            
    def clear_all(self):
        """清理所有数据"""
        self.document_processor.clear_all()
        
        if os.path.exists(self.vector_store_dir):
            shutil.rmtree(self.vector_store_dir)
            logger.info("向量数据库已清理")
        os.makedirs(self.vector_store_dir, exist_ok=True)
        logger.info("向量数据库目录已重新创建")