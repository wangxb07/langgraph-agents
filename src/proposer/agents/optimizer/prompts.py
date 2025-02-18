OPTIMIZER_SYSTEM_PROMPT = """您是提案优化专家。基于评估历史和改进建议，生成一个优化后的提案版本。"""

OPTIMIZER_ANALYSIS_PROMPT = """请分析以下提案、评估结果和历史改进记录：

# 当前提案
{current_proposal}

# 评估结果
{evaluation_results}

# 历史改进记录
{improvement_history}

请提供以下分析：
1. 识别关键问题：找出评分较低的维度和主要问题点
2. 汇总改进建议：整理各维度的具体改进建议
3. 总结历史经验：提取历史改进中的成功经验和需要避免的问题
"""

OPTIMIZER_OPTIMIZATION_PROMPT = """请基于以下信息优化提案：

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
4. 确保改进不会引入新问题

生成一个优化后的提案版本。"""