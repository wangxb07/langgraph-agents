import os
import logging
from typing import List, Dict, Any, Optional, Callable
from langchain.schema import Document
from qcloud_cos import CosConfig, CosS3Client
from langchain_community.document_loaders import PLACEHOLDER_FOR_SECRET_ID
from langchain_text_splitters import PLACEHOLDER_FOR_SECRET_ID

logger = logging.getLogger(__name__)

class PLACEHOLDER_FOR_SECRET_ID:
    """基于腾讯云 COS 的 DocumentProcessor 实现"""
    
    def __init__(self, secret_id: str, secret_key: str, region: str, bucket: str):
        """初始化 COS 客户端
        
        Args:
            secret_id: COS 的 Secret ID
            secret_key: COS 的 Secret Key
            region: COS 存储桶所在区域（如 'ap-guangzhou'）
            bucket: COS 存储桶名称
        """
        # 初始化COS配置
        self.config = CosConfig(
            Region=region,
            Secret_id=secret_id,
            Secret_key=secret_key,
            Scheme="https"
        )
        self.client = CosS3Client(self.config)
        self.bucket = bucket
        self.base_prefix = "documents/"  # COS 中的文档存储前缀
        
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
        if not processing_callback:
            raise ValueError("必须提供 processing_callback 函数")
        if not callable(processing_callback):
            raise ValueError("processing_callback 必须是可调用函数")
            
        try:
            # 使用 PLACEHOLDER_FOR_SECRET_ID 加载文档
            loader = PLACEHOLDER_FOR_SECRET_ID(
                conf=self.config,
                bucket=self.bucket,
                prefix=self.base_prefix
            )
            
            # 加载文档
            docs = loader.load()
            logger.info(f"从COS加载了 {len(docs)} 个文档")
            
            # 添加自定义元数据
            for doc in docs:
                # 从source中提取Key，因为PLACEHOLDER_FOR_SECRET_ID可能没有直接提供Key
                source = doc.metadata.get('source', '')
                key = source.replace(f'cos://{self.bucket}/', '')
                if not key:
                    logger.warning(f"跳过没有有效Key的文档: {source}")
                    continue
                
                logger.info(f"处理文档: {key}")
                logger.debug(f"文档元数据: {doc.metadata}")

                # 设置基本元数据 - 确保所有值都是有效类型（str, int, float, bool）
                # 为了避免中文文件名可能导致的问题，为每个文档生成一个唯一ID
                doc_id = f"doc_{os.urandom(4).hex()}"
                
                # 从文件名中提取标题，处理可能的中文字符
                filename = os.path.basename(key)
                title = filename.replace('.md', '').replace('.txt', '')
                
                doc.metadata.update({
                    'id': doc_id,
                    'Key': key,
                    'source': f"cos://{self.bucket}/{key}",
                    'type': 'markdown' if key.endswith('.md') else 'text',
                    'bucket': self.bucket,
                    'created_at': "unknown",  # 使用字符串而不是None
                    'title': title,
                    'filename': filename
                })
                
                # 清理元数据中的None值和可能导致问题的值
                for k in list(doc.metadata.keys()):
                    if doc.metadata[k] is None:
                        doc.metadata[k] = "unknown"
                    # 确保所有值都是基本类型
                    if not isinstance(doc.metadata[k], (str, int, float, bool)):
                        try:
                            doc.metadata[k] = str(doc.metadata[k])
                        except:
                            doc.metadata[k] = "unknown"
            
            headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
            text_splitter = PLACEHOLDER_FOR_SECRET_ID(headers_to_split_on=headers_to_split_on)

            total_chunks = 0
            for doc in docs:
                try:
                    splits = text_splitter.split_text(doc.page_content)
                    total_chunks += len(splits)
                    
                    # 确保每个分割后的文档都有正确的元数据
                    for split in splits:
                        # 复制元数据并确保没有None值
                        clean_metadata = {}
                        for k, v in doc.metadata.items():
                            if v is None:
                                clean_metadata[k] = "unknown"
                            else:
                                clean_metadata[k] = v
                        
                        split.metadata.update(clean_metadata)
                    
                    processing_callback(splits)
                    
                    if splits:
                        logger.info(f"文档片段示例内容: {splits[0].page_content[:100]}...")

                    logger.info(f"已处理文档片段：{doc.metadata['source']}, 生成 {len(splits)} 个片段")
                except Exception as e:
                    logger.error(f"处理文档失败 {doc.metadata.get('source', 'unknown')}: {str(e)}")
                    continue
                    
            logger.info(f"文档处理完成，共处理 {total_chunks} 个文档片段")
        except Exception as e:
            logger.error(f"加载文档时发生错误: {str(e)}", exc_info=True)
            raise
    
    def clear_all(self) -> None:
        """清空 COS 中的所有文档"""
        logger.info(f"清空 COS 存储桶 {self.bucket} 中的文档")
        marker = ""
        while True:
            response = self.client.list_objects(
                Bucket=self.bucket,
                Prefix=self.base_prefix,
                Marker=marker
            )
            contents = response.get('Contents', [])
            for obj in contents:
                self.client.delete_object(Bucket=self.bucket, Key=obj['Key'])
            if not response.get('IsTruncated', False):
                break
            marker = response['NextMarker']
    
    def add_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """添加文档到 COS
        
        Args:
            content: 文档内容
            metadata: 文档元数据
        
        Returns:
            文档ID（COS 中的对象键）
        """
        doc_id = f"{self.base_prefix}{metadata.get('title', 'doc')}_{os.urandom(8).hex()}"
        self.client.put_object(
            Bucket=self.bucket,
            Key=doc_id,
            Body=content.encode('utf-8'),
            Metadata=metadata
        )
        logger.info(f"添加文档: {doc_id}")
        return doc_id
    
    def remove_document(self, doc_id: str) -> None:
        """从 COS 中删除文档"""
        self.client.delete_object(Bucket=self.bucket, Key=doc_id)
        logger.info(f"删除文档: {doc_id}")
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """从 COS 中获取文档"""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=doc_id)
            content = response['Body'].get_raw_stream().read().decode('utf-8')
            metadata = response.get('Metadata', {})
            return {"content": content, "metadata": metadata}
        except Exception as e:
            logger.warning(f"获取文档失败 {doc_id}: {e}")
            return None
    
    def search_documents(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Document]:
        """搜索 COS 中的文档
        
        注意: COS 不支持原生搜索，此处仅为示例，实际需结合向量存储
        """
        logger.info(f"搜索文档: {query}")
        results = []
        marker = ""
        while len(results) < top_k:
            response = self.client.list_objects(
                Bucket=self.bucket,
                Prefix=self.base_prefix,
                Marker=marker
            )
            contents = response.get('Contents', [])
            for obj in contents:
                doc = self.get_document(obj['Key'])
                if doc and (not filter_dict or all(k in doc['metadata'] and doc['metadata'][k] == v for k, v in filter_dict.items())):
                    results.append(Document(page_content=doc['content'], metadata=doc['metadata']))
                if len(results) >= top_k:
                    break
            if not response.get('IsTruncated', False):
                break
            marker = response['NextMarker']
        return results[:top_k]