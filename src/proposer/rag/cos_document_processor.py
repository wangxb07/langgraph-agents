import os
import logging
import uuid
import tempfile
from typing import List, Dict, Any, Callable, Optional
from langchain.schema import Document
from qcloud_cos import CosConfig, CosS3Client
from langchain_text_splitters import PLACEHOLDER_FOR_SECRET_ID, PLACEHOLDER_FOR_SECRET_ID
from .pdf_processor import PDFProcessor

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
        
    def _process_file(self, key: str, content: bytes) -> List[Document]:
        """处理单个文件
        
        Args:
            key: 文件名
            content: 文件内容
            
        Returns:
            Document列表
        """
        # 根据文件扩展名判断类型
        ext = os.path.splitext(key)[1].lower()
        
        if ext == '.pdf':
            # 处理PDF文件
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(content)
                temp_file.flush()
                try:
                    documents = PDFProcessor.extract_text_from_pdf(temp_file.name)
                    # 添加通用元数据
                    for doc in documents:
                        doc.metadata.update({
                            'Key': key,
                            'source': f"cos://{self.bucket}/{key}",
                            'bucket': self.bucket,
                            'created_at': "unknown",
                            'title': os.path.basename(key)
                        })
                    return documents
                finally:
                    # 清理临时文件
                    os.unlink(temp_file.name)
        else:
            # 处理文本文件
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('gbk', errors='ignore')
            
            # 根据文件类型选择不同的分割器
            splits = []
            
            # 对Markdown文件使用PLACEHOLDER_FOR_SECRET_ID
            if ext == '.md':
                logger.info(f"使用PLACEHOLDER_FOR_SECRET_ID处理Markdown文件: {key}")
                
                # 定义Markdown标题层级
                headers_to_split_on = [
                    ("#", "header_1"),
                    ("##", "header_2"),
                    ("###", "header_3"),
                    ("####", "header_4"),
                ]
                
                # 使用Markdown专用分割器
                md_splitter = PLACEHOLDER_FOR_SECRET_ID(headers_to_split_on=headers_to_split_on)
                md_splits = md_splitter.split_text(text)
                
                # 如果Markdown分割器没有产生分割（可能没有标题），则使用通用分割器作为后备
                if md_splits:
                    splits = md_splits
                else:
                    logger.info(f"Markdown文件没有标题，使用PLACEHOLDER_FOR_SECRET_ID作为后备: {key}")
                    text_splitter = PLACEHOLDER_FOR_SECRET_ID(
                        chunk_size=1800,
                        chunk_overlap=200,
                        length_function=len,
                        separators=["\n\n", "\n", ". ", "? ", "! ", "；", "。", " ", ""]
                    )
                    splits = text_splitter.create_documents([text])
            else:
                # 对其他文本文件使用通用分割器
                logger.info(f"使用PLACEHOLDER_FOR_SECRET_ID处理文本文件: {key}")
                text_splitter = PLACEHOLDER_FOR_SECRET_ID(
                    chunk_size=1800,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", "? ", "! ", "；", "。", " ", ""]
                )
                splits = text_splitter.create_documents([text])
            
            # 为每个分割添加元数据
            for i, split in enumerate(splits):
                split.metadata.update({
                    'Key': key,
                    'source': f"cos://{self.bucket}/{key}",
                    'type': 'markdown' if ext == '.md' else 'text',
                    'bucket': self.bucket,
                    'created_at': "unknown",
                    'title': os.path.basename(key),
                    'chunk_id': i,
                    'chunk_count': len(splits)
                })
            
            # 如果没有分割（文本很短），则创建一个文档
            if not splits:
                logger.info(f"文件内容很短，不需要分割: {key}")
                doc = Document(
                    page_content=text,
                    metadata={
                        'Key': key,
                        'source': f"cos://{self.bucket}/{key}",
                        'type': 'markdown' if ext == '.md' else 'text',
                        'bucket': self.bucket,
                        'created_at': "unknown",
                        'title': os.path.basename(key)
                    }
                )
                return [doc]
            
            return splits
    
    def load_and_index_files(
        self,
        processing_callback: Callable[[List[Document]], None]
    ) -> None:
        """加载并索引COS中的文档
        
        Args:
            processing_callback: 文档处理回调函数
        """
        try:
            # 列出所有文档
            response = self.client.list_objects(
                Bucket=self.bucket,
                Prefix=self.base_prefix
            )
            
            if 'Contents' not in response:
                logger.warning(f"No documents found in bucket {self.bucket}")
                return
            
            # 批量处理文档
            batch_size = 10
            current_batch = []
            
            for item in response['Contents']:
                key = item['Key']
                
                # 跳过目录
                if key.endswith('/'):
                    continue
                    
                # 获取文件内容
                try:
                    response = self.client.get_object(
                        Bucket=self.bucket,
                        Key=key
                    )
                    content = response['Body'].get_raw_stream().read()
                    
                    # 处理文件
                    documents = self._process_file(key, content)
                    current_batch.extend(documents)
                    
                    # 达到批处理大小时处理
                    if len(current_batch) >= batch_size:
                        processing_callback(current_batch)
                        current_batch = []
                        
                except Exception as e:
                    logger.error(f"处理文件 {key} 时出错: {str(e)}")
                    continue
            
            # 处理剩余的文档
            if current_batch:
                processing_callback(current_batch)
                
        except Exception as e:
            logger.error(f"加载文档时出错: {str(e)}")
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