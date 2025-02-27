PROPOSER_SYSTEM_PROMPT = """您是提案生成专家。基于输入生成一个新的提案。
"""

PROPOSER_BASE_PROMPT = """基于以下背景，生成详细的创新方案：
{input}

请确保内容尽量详细。
"""

PROPOSER_RAG_PROMPT = """基于以下背景和参考资料，生成详细的创新方案：
{input}

# 参考资料
{references_text}

要求：
1. 充分利用参考资料中的相关信息
"""