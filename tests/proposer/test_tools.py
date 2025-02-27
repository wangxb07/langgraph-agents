import pytest
import logging
from unittest.mock import patch, MagicMock
from langchain.schema import Document
from proposer.tools import rag_retrieve, get_rag_tool
from proposer.rag.rag import RAGTool

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestRAGTools:
    """RAG工具测试类"""
    
    @patch('proposer.tools.get_rag_tool')
    def test_rag_retrieve_success(self, mock_get_rag_tool):
        """测试成功检索文档的情况"""
        # 创建模拟文档
        mock_docs = [
            Document(
                page_content="这是第一个测试文档的内容",
                metadata={"source": "cos://test-bucket/doc1.txt", "type": "text"}
            ),
            Document(
                page_content="这是第二个测试文档的内容",
                metadata={"source": "cos://test-bucket/doc2.txt", "type": "text"}
            )
        ]
        
        # 创建模拟RAG工具
        mock_rag_tool = MagicMock(spec=RAGTool)
        mock_rag_tool.retrieve.return_value = mock_docs
        
        # 设置get_rag_tool返回模拟工具
        mock_get_rag_tool.return_value = mock_rag_tool
        
        # 执行测试
        result = rag_retrieve("测试查询")
        
        # 验证结果
        assert "文档 1" in result
        assert "这是第一个测试文档的内容" in result
        assert "文档 2" in result
        assert "这是第二个测试文档的内容" in result
        
        # 验证调用
        mock_rag_tool.retrieve.assert_called_once_with("测试查询")
    
    @patch('proposer.tools.get_rag_tool')
    def test_rag_retrieve_empty_results(self, mock_get_rag_tool):
        """测试检索结果为空的情况"""
        # 创建模拟RAG工具
        mock_rag_tool = MagicMock(spec=RAGTool)
        mock_rag_tool.retrieve.return_value = []
        
        # 设置get_rag_tool返回模拟工具
        mock_get_rag_tool.return_value = mock_rag_tool
        
        # 执行测试
        result = rag_retrieve("不存在的查询")
        
        # 验证结果
        assert "未找到相关文档" in result
        
        # 验证调用
        mock_rag_tool.retrieve.assert_called_once_with("不存在的查询")
    
    @patch('proposer.tools.get_rag_tool')
    def test_rag_retrieve_error_handling(self, mock_get_rag_tool):
        """测试错误处理"""
        # 创建模拟RAG工具
        mock_rag_tool = MagicMock(spec=RAGTool)
        mock_rag_tool.retrieve.side_effect = Exception("测试异常")
        
        # 设置get_rag_tool返回模拟工具
        mock_get_rag_tool.return_value = mock_rag_tool
        
        # 执行测试
        result = rag_retrieve("触发异常的查询")
        
        # 验证结果
        assert "执行文档检索时出错" in result
        assert "测试异常" in result
        
        # 验证调用
        mock_rag_tool.retrieve.assert_called_once_with("触发异常的查询")
    
    @patch('proposer.tools.get_rag_tool')
    def test_rag_retrieve_integration(self, mock_get_rag_tool, caplog):
        """集成测试：测试实际RAG工具的行为"""
        # 使用真实的RAG工具（可选，取决于环境）
        # 如果不想使用真实工具，可以注释掉这个测试或继续使用模拟
        
        # 创建模拟文档，模拟更复杂的内容
        mock_docs = [
            Document(
                page_content="人工智能（Artificial Intelligence，AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
                metadata={"source": "cos://test-bucket/ai_intro.txt", "type": "text"}
            ),
            Document(
                page_content="机器学习是人工智能的一个分支，是一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、计算复杂性理论等多门学科。",
                metadata={"source": "cos://test-bucket/machine_learning.txt", "type": "text"}
            ),
            Document(
                page_content="深度学习是机器学习的分支，是一种以人工神经网络为架构，对数据进行表征学习的算法。",
                metadata={"source": "cos://test-bucket/deep_learning.txt", "type": "text"}
            )
        ]
        
        # 创建模拟RAG工具
        mock_rag_tool = MagicMock(spec=RAGTool)
        mock_rag_tool.retrieve.return_value = mock_docs
        
        # 设置get_rag_tool返回模拟工具
        mock_get_rag_tool.return_value = mock_rag_tool
        
        # 执行测试，使用更复杂的查询
        with caplog.at_level(logging.INFO):
            result = rag_retrieve("人工智能和机器学习的关系")
        
        # 验证结果包含所有文档
        assert "人工智能" in result
        assert "机器学习是人工智能的一个分支" in result
        assert "深度学习是机器学习的分支" in result
        
        # 验证文档格式
        assert "文档 1:" in result
        assert "文档 2:" in result
        assert "文档 3:" in result
        
        # 验证调用
        mock_rag_tool.retrieve.assert_called_once_with("人工智能和机器学习的关系")
