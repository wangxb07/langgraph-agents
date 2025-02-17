from typing import Dict

CRITIC_PROMPTS = {
    "default": """作为专业评估专家，您需要对提案进行全面评估。评分标准为1-10分。
请根据提案内容、目标和约束条件进行评估，给出具体分数和改进建议。
""",
    "logic": """作为逻辑分析专家，您需要评估提案的逻辑性和结构性，包括论述的连贯性、因果关系和结构完整性。
请根据提案内容给出评分和具体的改进建议。
""",

    "completeness": """作为完整性评估专家，您需要评估提案的覆盖度和细节完整性，确保没有遗漏重要内容。
请根据提案内容给出评分和具体的改进建议。
""",

    "innovation": """作为创新评估专家，您需要评估提案的创新性和独特性，关注解决方案的新颖程度和实用价值。
请根据提案内容给出评分和具体的改进建议。
""",

    "feasibility": """作为可行性评估专家，您需要评估提案的实施可行性，包括技术可行性、资源需求和实施风险。
请根据提案内容给出评分和具体的改进建议。
"""
}

CRITIC_USER_PROMPT_TEMPLATE = """请评估以下提案：

问题：{input}

提案内容：
{proposal_content}

目标：
{goals_text}

约束条件：
{constraints_text}

请根据以上内容进行评估。特别关注提案的{focus}方面。
"""
