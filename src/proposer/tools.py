"""This module provides example tools for web scraping and search functionality.

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Dict, List, Optional
import os
from dotenv import load_dotenv

from langchain_core.tools import tool
from rag.rag import RAGTool
from rag.cos_document_processor import TencentCOSDocumentProcessor

# 创建 RAG 工具实例
_rag_tool = None

load_dotenv()

def get_rag_tool(prompt_template: Optional[str] = None) -> RAGTool:
    """获取或创建 RAG 工具实例
    
    Args:
        prompt_template: 可选的提示模板
        
    Returns:
        RAG 工具实例
    """
    global _rag_tool
    
    if _rag_tool is None:
        if prompt_template is None:
            prompt_template = """你是一个专业的助手。请基于以下参考文档回答用户的问题。

参考文档:
{context}

用户问题: {question}

要求：
1. 请基于参考文档提供准确、相关的回答
2. 如果参考文档中没有相关信息，请明确说明"抱歉，参考文档中没有相关信息"
3. 回答要简洁、专业，并确保信息的准确性
4. 如果需要引用具体内容，请说明出处

请回答："""
        
        # 获取环境变量中的 Tencent COS 凭证
        secret_id = os.getenv("TENCENT_COS_SECRET_ID")
        secret_key = os.getenv("TENCENT_COS_SECRET_KEY")
        region = os.getenv("TENCENT_COS_REGION", "ap-shanghai")
        bucket = os.getenv("TENCENT_COS_BUCKET", "rag-1309172277")
        
        # 检查必要的凭证是否存在
        if not secret_id or not secret_key:
            raise ValueError("TENCENT_COS_SECRET_ID 和 TENCENT_COS_SECRET_KEY 环境变量必须设置")
        
        # 创建 RAG 工具实例
        _rag_tool = RAGTool(
            prompt_template=prompt_template,
            document_processor=TencentCOSDocumentProcessor(
                secret_id=secret_id,
                secret_key=secret_key,
                region=region,
                bucket=bucket
            )
        )
    
    return _rag_tool

@tool
def rag_search(query: str) -> str:
    """从知识库中搜索并回答问题。
    
    Args:
        query: 要搜索的查询文本
        
    Returns:
        基于知识库的回答
    """
    try:
        rag_tool = get_rag_tool()
        return rag_tool.query(query)
    except Exception as e:
        return f"执行 RAG 搜索时出错: {str(e)}"

@tool
def rag_retrieve(query: str) -> str:
    """从知识库中仅检索相关文档，不生成回答。
    
    Args:
        query: 要搜索的查询文本
        
    Returns:
        检索到的相关文档内容
    """
    try:
        rag_tool = get_rag_tool()
        documents = rag_tool.retrieve(query)
        if not documents:
            return "未找到相关文档"
        
        results = []
        for i, doc in enumerate(documents, 1):
            results.append(f"文档 {i}:\n{doc.page_content}\n")
        
        return "\n".join(results)
    except Exception as e:
        return f"执行文档检索时出错: {str(e)}"

# 导出工具列表
TOOLS = [rag_search, rag_retrieve]
