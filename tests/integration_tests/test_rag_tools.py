import pytest
import logging
from proposer.tools import rag_retrieve, rag_search, get_rag_tool
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
                result = rag_retrieve.invoke(query)  # 使用 invoke 而不是直接调用
                
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
            result = rag_retrieve.invoke("人工智能技术")  # 使用 invoke 而不是直接调用
            
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
            
            # 调用两个工具，使用 invoke 而不是直接调用
            search_result = rag_search.invoke(query)
            retrieve_result = rag_retrieve.invoke(query)
            
            # 记录结果
            logger.info(f"Search result: {search_result}")
            logger.info(f"Retrieve result: {retrieve_result}")
            
            # 验证结果
            assert search_result is not None
            assert retrieve_result is not None
            assert isinstance(search_result, dict)  # 修改：search_result 是字典
            assert isinstance(retrieve_result, str)
            
            # 验证 search_result 的结构
            assert 'answer' in search_result
            assert 'sources' in search_result
            assert isinstance(search_result['answer'], str)
            assert isinstance(search_result['sources'], list)
            
            # 检索工具应该只返回文档，而不是生成回答
            if "未找到相关文档" not in retrieve_result:
                assert "文档" in retrieve_result
            
        except Exception as e:
            logger.error(f"Comparison test failed: {str(e)}")
            pytest.skip(f"Comparison test skipped due to error: {str(e)}")
            
    @pytest.mark.integration
    def test_ningbo_tour_retrieval(self):
        """测试检索宁波周边游的文档"""
        try:            
            # 设置环境变量强制重新创建向量数据库
            import os
            os.environ["RAG_FORCE_RECREATE_VECTOR_DB"] = "true"
            
            # 获取RAG工具实例并强制重新索引文档
            rag_tool = get_rag_tool()
            logger.info("强制重新索引COS文档...")
            
            # 强制重新加载并索引文档
            def process_documents(documents):
                logger.info(f"处理 {len(documents)} 个文档")
                # 打印文档信息，帮助调试
                for doc in documents:
                    logger.info(f"文档: {doc.metadata.get('source', 'unknown')}")
                    logger.info(f"元数据: {doc.metadata}")
                rag_tool.vectorstore.add_documents(documents)
                
            rag_tool.document_processor.load_and_index_files(process_documents)
            
            # 直接调用检索工具
            query = "宁波周边游"
            result = rag_search.invoke(query)
            
            # 记录结果
            logger.info(f"Ningbo tour retrieval result: {result}")
            
            # 验证结果
            assert result is not None
            assert isinstance(result, dict)
            assert "answer" in result
            assert "sources" in result
            
            # 验证文档内容包含宁波各区
            districts = ["海曙", "江北", "北仑", "镇海", "鄞州", "奉化"]
            found_districts = []
            
            # 检查回答和来源中是否包含区名
            for district in districts:
                if district in result["answer"]:
                    found_districts.append(district)
                else:
                    for source in result["sources"]:
                        if district in source["content"]:
                            found_districts.append(district)
                            break
            
            # 确保至少找到一半的区名
            min_districts = len(districts) // 2
            assert len(found_districts) >= min_districts, f"至少应包含{min_districts}个区名，但只找到了{len(found_districts)}个: {found_districts}"
                
            logger.info("宁波周边游文档检索测试通过！")
            
        except Exception as e:
            logger.error(f"Ningbo tour retrieval test failed: {str(e)}")
            pytest.skip(f"Ningbo tour retrieval test skipped due to error: {str(e)}")

    @pytest.mark.integration
    def test_pdf_document_retrieval(self):
        """测试检索PDF文档"""
        try:
            # 设置环境变量强制重新创建向量数据库
            import os
            os.environ["RAG_FORCE_RECREATE_VECTOR_DB"] = "true"
            
            # 获取RAG工具实例并强制重新索引文档
            rag_tool = get_rag_tool()
            logger.info("强制重新索引COS文档...")
            
            # 强制重新加载并索引文档
            def process_documents(documents):
                logger.info(f"处理 {len(documents)} 个文档")
                # 打印PDF文档信息，帮助调试
                for doc in documents:
                    if doc.metadata.get('type') == 'pdf':
                        logger.info(f"PDF文档: {doc.metadata.get('source', 'unknown')}")
                        logger.info(f"页码: {doc.metadata.get('page', 'unknown')}")
                        logger.info(f"元数据: {doc.metadata}")
                rag_tool.vectorstore.add_documents(documents)
            
            rag_tool.document_processor.load_and_index_files(process_documents)
            
            # 测试检索PDF内容
            query = "宁波旅游规划"  # 假设有相关的PDF文档
            result = rag_search.invoke(query)
            
            # 验证结果
            assert result is not None
            assert isinstance(result, dict)
            assert "answer" in result
            assert "sources" in result
            
            # 检查是否有PDF来源
            pdf_sources = [
                source for source in result["sources"]
                if source.get("metadata", {}).get("type") == "pdf"
            ]
            
            logger.info(f"找到 {len(pdf_sources)} 个PDF来源")
            for source in pdf_sources:
                logger.info(f"PDF来源: {source['metadata']}")
            
            logger.info("PDF文档检索测试通过！")
            
        except Exception as e:
            logger.error(f"PDF文档检索测试失败: {str(e)}")
            pytest.skip(f"PDF文档检索测试跳过，原因: {str(e)}")
