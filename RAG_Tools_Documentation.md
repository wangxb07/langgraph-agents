# RAG 工具使用文档

## 概述

本文档提供了有关使用检索增强生成（RAG）工具进行文档检索和问答的指南。RAG 工具旨在通过将文档检索与生成式响应相结合来增强模型的能力。

## 组件

### 1. RAGTool

- **用途**：管理文档的检索和基于给定提示模板的答案生成。
- **初始化**：
  - 使用腾讯云 COS 进行文档存储和检索。
  - 需要环境变量来获取凭证：
    - `TENCENT_COS_SECRET_ID`
    - `TENCENT_COS_SECRET_KEY`
    - `TENCENT_COS_REGION`
    - `TENCENT_COS_BUCKET`
  - 提供了示例提示模板以指导响应生成过程。

### 2. 函数

- **`get_rag_tool(prompt_template: Optional[str] = None) -> RAGTool`**:
  - 使用指定的提示模板或默认模板获取或创建 RAG 工具实例。

- **`rag_search(query: str) -> str`**:
  - 从知识库中搜索并生成基于检索到的文档的答案。
  - 返回包含生成答案的字符串或搜索失败时的错误信息。

- **`rag_retrieve(query: str) -> str`**:
  - 从知识库中检索相关文档而不生成答案。
  - 返回检索到的文档内容或检索失败时的错误信息。

## 集成测试

### 测试套件：`TestRAGTools`

- **用途**：验证 RAG 工具与自定义聊天模型的集成和功能。
- **测试**：
  - `test_rag_retrieve_with_model`：测试 RAG 检索工具与自定义聊天模型的集成。
  - `test_direct_rag_retrieve_call`：直接测试 RAG 检索工具。
  - `test_rag_search_and_retrieve_comparison`：比较搜索和检索工具的输出。
  - `test_ningbo_tour_retrieval`：测试与宁波旅游相关的文档检索。
  - `test_pdf_document_retrieval`：测试 PDF 文档的检索。

## 使用示例

### 环境设置

确保已设置腾讯云 COS 凭证所需的环境变量。使用 `.env` 文件安全管理这些变量。

### 工具调用

```python
from proposer.tools import rag_search, rag_retrieve

# 搜索并生成答案
query = "人工智能的应用"
answer = rag_search(query)
print("生成的答案：", answer)

# 仅检索文档
documents = rag_retrieve(query)
print("检索到的文档：", documents)
```

### 集成

将 RAG 工具绑定到自定义聊天模型以增强功能。利用集成测试验证工具的设置和功能。

## 安全实践

- 将敏感凭证存储在环境变量中。
- 确保 `.env` 文件被 gitignore 以防止凭证意外泄露。
