OPTIMIZER_SYSTEM_PROMPT = """您是提案优化专家。基于评估历史和改进建议，生成一个优化后的提案版本。"""

OPTIMIZER_OPTIMIZATION_PROMPT = """请基于以下信息优化提案：

# 当前提案
{current_proposal}

# 评估结果
{evaluation_results}

# 历史改进记录
{improvement_history}

# 参考资料
{references}

请重点关注：
1. 评分较低的部分
2. 具体的改进建议
3. 历史改进中的成功经验
4. 避免重复之前的问题

生成一个优化后的提案版本。"""