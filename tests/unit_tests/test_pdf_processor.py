"""测试PDF处理器"""
import os
import pytest
from proposer.rag.pdf_processor import PDFProcessor

@pytest.fixture
def sample_pdf_path(tmp_path):
    """创建一个示例PDF文件"""
    # 使用PyMuPDF创建一个简单的PDF文件
    import fitz
    doc = fitz.open()
    
    # 第一页
    page = doc.new_page()
    text = "测试文本第一页"
    # 使用insert_text代替，设置字体支持中文
    page.insert_text(
        (50, 50),  # 位置
        text,  # 文本
        fontname="china-s",  # 中文字体
        fontsize=12,  # 字体大小
        color=(0, 0, 0)  # 黑色
    )
    
    # 第二页
    page = doc.new_page()
    text = "测试文本第二页"
    page.insert_text(
        (50, 50),
        text,
        fontname="china-s",
        fontsize=12,
        color=(0, 0, 0)
    )
    
    # 保存PDF文件
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()
    
    yield str(pdf_path)
    
    # 清理
    os.unlink(str(pdf_path))

def test_extract_text_from_pdf(sample_pdf_path):
    """测试从PDF中提取文本"""
    # 提取文本
    documents = PDFProcessor.extract_text_from_pdf(sample_pdf_path)
    
    # 验证基本属性
    assert len(documents) == 2  # 应该有两页
    assert all(doc.page_content for doc in documents)  # 每页都应该有内容
    
    # 验证元数据
    for i, doc in enumerate(documents):
        assert doc.metadata["source"] == sample_pdf_path
        assert doc.metadata["page"] == i + 1
        assert doc.metadata["total_pages"] == 2
        assert doc.metadata["type"] == "pdf"
        
    # 验证内容
    assert "测试文本第一页" in documents[0].page_content
    assert "测试文本第二页" in documents[1].page_content

def test_extract_text_from_nonexistent_pdf():
    """测试处理不存在的PDF文件"""
    with pytest.raises(Exception) as exc_info:
        PDFProcessor.extract_text_from_pdf("nonexistent.pdf")
    assert "处理PDF文件时出错" in str(exc_info.value)
