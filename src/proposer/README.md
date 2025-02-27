# 提案工作流 (Proposal Workflow)

提案工作流是一个基于LangGraph的工作流系统，用于生成、评估和优化提案内容。

## 主要特点

- **可配置的代理模型**: 可以为提案生成器、评估器和优化器分别指定不同的模型
- **灵活的评估重点**: 支持多种评估重点，如逻辑性、完整性、创新性和可行性
- **可配置的迭代参数**: 可以设置最大迭代次数和优秀评分标准
- **用户交互**: 支持用户在评估阶段提供反馈和决策

## 工作流程

提案工作流包含以下步骤：

1. **初始化代理**：根据配置初始化提案生成器、多维度评估器和优化器
2. **生成提案**：根据输入、目标和约束条件生成初始提案
3. **多维度评估**：从逻辑性、完整性、创新性和可行性四个维度对提案进行评估
4. **仲裁**：根据评估结果决定是继续优化还是结束工作流
5. **优化**：如果需要继续优化，则根据评估结果对提案进行优化，然后返回步骤3

## 多维度评估

提案工作流使用多个评估代理从不同维度对提案进行全面评估：

- **逻辑性评估**：评估提案的逻辑性和结构性，包括论述的连贯性、因果关系和结构完整性
- **完整性评估**：评估提案的覆盖度和细节完整性，确保没有遗漏重要内容
- **创新性评估**：评估提案的创新性和独特性，关注解决方案的新颖程度和实用价值
- **可行性评估**：评估提案的实施可行性，包括技术可行性、资源需求和实施风险

每个维度的评估结果会被整合为一个综合评分，并提供详细的改进建议。

## 架构设计

- **工作流类 (ProposalWorkflow)**: 负责协调整个提案生成、评估和优化过程
  - 包含提案生成器、评估器和优化器实例
  - 通过图节点函数实现工作流逻辑
- **状态类 (ProposalState)**: 存储工作流执行过程中的状态信息
  - 包含提案内容、评估结果、迭代次数等
  - 不包含代理实例，代理实例存储在工作流类中

## 配置参数

工作流支持以下配置参数：

- `model`: 默认模型，用于未指定特定代理模型时
- `proposer_model`: 提案生成器使用的模型
- `critic_model`: 评估器使用的模型
- `optimizer_model`: 优化器使用的模型
- `max_iterations`: 最大迭代次数
- `excellent_score`: 优秀评分标准，达到此分数时工作流结束

## 使用示例

```python
from src.proposer.graph import create_graph, ProposalInput
import uuid

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

# 创建初始状态
initial_state = ProposalInput(
    input="如何提高团队协作效率？",
    constraints=[
        {"type": "时间", "value": "方案应该能在1个月内实施完成"},
        {"type": "预算", "value": "实施成本不应超过5000元"},
        {"type": "技术", "value": "不需要复杂的技术支持，普通团队成员可以理解和使用"}
    ],
    goals=[
        "提出可行的团队协作改进方案",
        "方案应该易于实施且成本较低",
        "方案应该能够适应远程和现场办公场景"
    ]
)

# 运行工作流
result = await workflow.ainvoke(initial_state.__dict__, config=config)

# 获取最终提案
final_proposal = result['proposals'][-1]

# 获取多维度评估结果
final_evaluation = result['evaluations'][-1]
overall_score = final_evaluation['score']
dimensions = final_evaluation['dimensions']  # 包含各维度的评估结果
suggestions = final_evaluation['suggestions']  # 包含各维度的改进建议
```

更多详细示例请参考 [examples/proposal_example.py](../../examples/proposal_example.py)。

## 注意事项

- 确保已安装所有必要的依赖
- 模型名称必须是系统支持的模型
- 评估重点必须是系统支持的类型之一
