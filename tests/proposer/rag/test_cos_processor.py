import pytest
import logging
from langchain.schema import Document
from proposer.rag.cos_document_processor import PLACEHOLDER_FOR_SECRET_ID

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def cos_processor():
    """创建COS处理器"""
    processor = PLACEHOLDER_FOR_SECRET_ID(
        secret_id="PLACEHOLDER_FOR_SECRET_ID",
        secret_key="PLACEHOLDER_FOR_SECRET_ID",
        region="ap-shanghai",
        bucket="rag-1309172277"
    )
    return processor

class PLACEHOLDER_FOR_SECRET_ID:
    """PLACEHOLDER_FOR_SECRET_ID 测试类"""
    def test_load_and_index_files_basic(self, cos_processor):
        """测试基本的文档加载和处理流程"""
        processed_docs = []
        processed_metadata = []
        
        def test_callback(splits):
            processed_docs.extend([d.page_content for d in splits])
            processed_metadata.extend([d.metadata for d in splits])
        
        # 执行测试
        cos_processor.load_and_index_files(processing_callback=test_callback)
        
        # 验证文档处理
        assert len(processed_docs) > 0
        # 验证元数据完整性
        assert all("source" in meta for meta in processed_metadata)
        assert all("type" in meta for meta in processed_metadata)
        assert all(isinstance(meta.get("source"), str) for meta in processed_metadata)
        assert all(meta.get("source").startswith("cos://") for meta in processed_metadata)
    
    def test_load_and_index_files_error_handling(self, cos_processor):
        """测试错误处理"""
        success_docs = []
        error_count = 0
        
        def test_callback(splits):
            nonlocal error_count
            try:
                success_docs.extend(splits)
            except Exception:
                error_count += 1
        
        # 执行测试 - 不应该因为单个文档处理失败而完全失败
        cos_processor.load_and_index_files(processing_callback=test_callback)
        
        # 验证至少有一些文档被成功处理
        assert len(success_docs) > 0
        
    def test_load_and_index_files_concurrent_safety(self, cos_processor):
        """测试并发安全性"""
        import threading
        
        processed_chunks = []
        lock = threading.Lock()
        
        def thread_safe_callback(splits):
            with lock:
                processed_chunks.extend(splits)
        
        # 创建多个线程同时处理
        threads = []
        for _ in range(3):
            thread = threading.Thread(
                target=cos_processor.load_and_index_files,
                args=(thread_safe_callback,)
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证处理结果 - 确保有数据被处理
        assert len(processed_chunks) > 0
