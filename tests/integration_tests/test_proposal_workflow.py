import pytest
from proposer.graph import ProposalWorkflow
from langgraph.graph import END


def test_create_proposal_graph():
    """测试创建提案工作流图"""
    # 创建工作流实例
    workflow = ProposalWorkflow(max_iterations=3)
    
    # 获取工作流图
    graph = workflow.create_graph()
    
    # 验证图的基本属性
    assert graph is not None
    assert graph.name == "LangGraph"
    
    # 验证图的节点
    assert hasattr(graph, "nodes")
    nodes = graph.nodes
    assert "propose" in nodes
    assert "evaluate" in nodes
    assert "arbitrate" in nodes
