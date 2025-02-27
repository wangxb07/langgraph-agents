"""
提案工作流示例

这个示例演示了如何使用提案工作流来生成、评估和优化提案。
工作流会使用多个评估代理从不同维度（逻辑性、完整性、创新性、可行性）对提案进行评估。
"""

import asyncio
import os
import sys
import uuid
from typing import Dict, List, Any

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.proposer.graph import create_graph
from src.proposer.graph import ProposalInput


async def main():
    """运行提案工作流示例"""
    
    # 创建工作流
    workflow = create_graph()
    
    # 配置参数
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),  # 添加线程ID
            "proposer_model": "qwen-max",  # 提案生成器使用的模型
            "critic_model": "qwen-plus",   # 评估器使用的模型
            "optimizer_model": "qwen-max", # 优化器使用的模型
            "max_iterations": 2,           # 最大迭代次数
            "excellent_score": 8.0         # 优秀评分标准
        }
    }
    
    # 定义输入参数
    input_text = "如何提高团队协作效率？"
    
    # 定义目标
    goals = [
        "提出可行的团队协作改进方案",
        "方案应该易于实施且成本较低",
        "方案应该能够适应远程和现场办公场景"
    ]
    
    # 定义约束条件
    constraints = [
        {"type": "时间", "value": "方案应该能在1个月内实施完成"},
        {"type": "预算", "value": "实施成本不应超过5000元"},
        {"type": "技术", "value": "不需要复杂的技术支持，普通团队成员可以理解和使用"}
    ]
    
    # 创建初始状态
    initial_state = ProposalInput(
        input=input_text,
        PLACEHOLDER_FOR_SECRET_ID,
        goals=goals
    )
    
    # 运行工作流
    result = await workflow.ainvoke(initial_state.__dict__, config=config)
    
    # 打印结果
    print("\n=== 提案工作流执行结果 ===")
    print(f"迭代次数: {result['iteration']}")
    print(f"最终状态: {result['status']}")
    
    # 打印最终提案
    print("\n=== 最终提案 ===")
    print(result['proposals'][-1])
    
    # 打印多维度评估结果
    print("\n=== 多维度评估结果 ===")
    final_evaluation = result['evaluations'][-1]
    print(f"综合评分: {final_evaluation['score']:.2f}")
    
    # 打印各维度评估
    print("\n各维度评估:")
    for focus, evaluation in final_evaluation['dimensions'].items():
        print(f"- {focus}: {evaluation['score']:.2f}")
        print(f"  评论: {evaluation['comments']}")
    
    # 打印改进建议
    print("\n改进建议:")
    for suggestion in final_evaluation['suggestions']:
        print(f"- {suggestion}")


if __name__ == "__main__":
    asyncio.run(main())
