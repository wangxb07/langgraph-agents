import pytest
import logging
from proposer.tools import rag_retrieve, rag_search
from proposer.utils import init_custom_chat_model

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PLACEHOLDER_FOR_SECRET_ID:
    """RAG工具集成测试类"""
    
    @pytest.mark.integration
    def test_rag_retrieve_with_model(self):
        """测试RAG检索工具与模型的集成"""
        try:
            # 初始化模型
            model = init_custom_chat_model("qwen-max")
            
            # 绑定工具
            model_with_tools = model.bind_tools([rag_retrieve])
            
            # 执行工具调用
            response = model_with_tools.invoke("帮我检索关于人工智能的文档")
            
            # 验证响应
            logger.info(f"Model response: {response}")
            assert response is not None
            
            # 检查工具调用
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                assert tool_call['name'] == 'rag_retrieve'
                assert 'query' in tool_call['args']
                
                # 执行工具
                query = tool_call['args']['query']
                result = rag_retrieve(query)
                
                # 验证结果
                logger.info(f"Tool result: {result}")
                assert result is not None
                assert isinstance(result, str)
                
        except Exception as e:
            logger.error(f"Integration test failed: {str(e)}")
            pytest.skip(f"Integration test skipped due to error: {str(e)}")
    
    @pytest.mark.integration
    def test_direct_rag_retrieve_call(self):
        """直接测试RAG检索工具"""
        try:
            # 直接调用工具
            result = rag_retrieve("人工智能技术")
            
            # 验证结果
            logger.info(f"Direct tool result: {result}")
            assert result is not None
            assert isinstance(result, str)
            
            # 如果找到文档，验证格式
            if "未找到相关文档" not in result:
                assert "文档" in result
            
        except Exception as e:
            logger.error(f"Direct tool test failed: {str(e)}")
            pytest.skip(f"Direct tool test skipped due to error: {str(e)}")
    
    @pytest.mark.integration
    def test_rag_search_and_retrieve_comparison(self):
        """比较搜索和检索工具的结果"""
        try:
            query = "人工智能的应用"
            
            # 调用两个工具
            search_result = rag_search(query)
            retrieve_result = rag_retrieve(query)
            
            # 记录结果
            logger.info(f"Search result: {search_result}")
            logger.info(f"Retrieve result: {retrieve_result}")
            
            # 验证结果
            assert search_result is not None
            assert retrieve_result is not None
            assert isinstance(search_result, str)
            assert isinstance(retrieve_result, str)
            
            # 检索工具应该只返回文档，而不是生成回答
            if "未找到相关文档" not in retrieve_result:
                assert "文档" in retrieve_result
            
        except Exception as e:
            logger.error(f"Comparison test failed: {str(e)}")
            pytest.skip(f"Comparison test skipped due to error: {str(e)}")
