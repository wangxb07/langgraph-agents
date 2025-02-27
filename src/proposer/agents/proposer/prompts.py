PROPOSER_SYSTEM_PROMPT = """您是提案生成专家。基于输入、目标和约束条件生成一个新的提案。
"""

PROPOSER_BASE_PROMPT = """基于以下背景，生成详细的创新方案：
{input}

# 目标
{goals_text}

# 约束条件
{constraints_text}

请确保内容尽量详细，并满足所有目标和约束条件。
"""

PROPOSER_RAG_PROMPT = """基于以下背景和参考资料，生成详细的创新方案：
{input}

# 目标
{goals_text}

# 约束条件
{constraints_text}

# 参考资料
{references_text}

要求：
1. 充分利用参考资料中的相关信息
2. 确保满足所有目标和约束条件
"""