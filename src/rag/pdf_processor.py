"""PDF文档处理模块"""
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from langchain_core.documents import Document

class PDFProcessor:
    """PDF文档处理器"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> List[Document]:
        """从PDF文件中提取文本
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            包含PDF内容的Document列表，每页生成一个Document
        """
        documents = []
        
        try:
            # 打开PDF文件
            pdf_document = fitz.open(file_path)
            
            # 遍历每一页
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # 提取文本，使用raw=True以获取更好的文本布局
                blocks = page.get_text("blocks", sort=True)
                text_blocks = []
                
                # 处理每个文本块
                for block in blocks:
                    if block[6] == 0:  # 0表示这是文本块
                        text_blocks.append(block[4])  # block[4]是文本内容
                
                # 合并文本块
                text = "\n".join(text_blocks)
                
                # 获取页面元数据
                metadata = {
                    "source": file_path,
                    "page": page_num + 1,
                    "total_pages": len(pdf_document),
                    "type": "pdf"
                }
                
                # 创建Document对象
                doc = Document(
                    page_content=text.strip(),
                    metadata=metadata
                )
                
                documents.append(doc)
            
            pdf_document.close()
            
        except Exception as e:
            raise Exception(f"处理PDF文件时出错: {str(e)}")
            
        return documents
