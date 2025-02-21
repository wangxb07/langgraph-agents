OPTIMIZER_SYSTEM_PROMPT = """您是提案优化专家。基于评估历史和改进建议，生成一个优化后的提案版本。"""

ANALYSIS_SYSTEM_PROMPT = """根据评估结果分析提案的关键问题和建议。"""

OPTIMIZER_ANALYSIS_PROMPT = """请分析评估结果：

# 评估结果
{evaluation_results}

请分析以下几个方面：
1. 主要问题和不足
2. 改进方向和建议
3. 需要优化的关键点

请给出结构化的分析结果。"""

OPTIMIZER_OPTIMIZATION_PROMPT = """基于以上分析，请优化提案。

# 当前提案
{current_proposal}

# 分析结果
{analysis_result}

# 参考资料
{references}

请根据分析结果中的key_issues和suggestions对提案进行优化。重点关注：
1. 针对每个关键问题进行改进
2. 采纳相关的具体建议
3. 参考成功经验

生成一个优化后的提案版本。"""