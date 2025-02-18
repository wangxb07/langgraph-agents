import pytest
from proposer.agents.critic.core import CriticAgent
from proposer.agents.optimizer.core import OptimizerAgent
from proposer.agents.proposer.core import ProposerAgent
from proposer.utils import init_custom_chat_model
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate

@pytest.fixture
def critic_agent():
    """创建一个CriticAgent实例用于测试"""
    return CriticAgent(model="qwen-plus")

@pytest.fixture
def optimizer_agent() -> OptimizerAgent:
    """创建一个OptimizerAgent实例用于测试"""
    return OptimizerAgent(model="qwen-plus")

@pytest.fixture
def proposer_agent():
    """创建一个ProposerAgent实例用于测试"""
    return ProposerAgent(model="qwen-plus")

def test_critic_agent_initialization(critic_agent):
    assert isinstance(critic_agent, CriticAgent)
    assert critic_agent.model is not None

def test_optimizer_agent_initialization(optimizer_agent):
    assert isinstance(optimizer_agent, OptimizerAgent)
    assert optimizer_agent.model is not None

def test_proposer_agent_initialization(proposer_agent):
    assert isinstance(proposer_agent, ProposerAgent)
    assert proposer_agent.model is not None

@pytest.mark.asyncio
async def test_critic_agent_evaluate_proposal(critic_agent):
    input_data = {
        "input": "test_input",
        "proposal_content": "test_proposal",
        "goals": ["test goal 1", "test goal 2"],
        "constraints": [{"type": "test", "value": "test"}]
    }
    response = await critic_agent.evaluate_proposal(**input_data)
    assert response is not None

@pytest.mark.asyncio
async def test_optimizer_agent_optimize(optimizer_agent):
    """测试优化器的optimize_proposal方法"""
    # 准备测试数据
    input_data = {
        "current_proposal": "这是一个测试提案，需要优化和改进。",
        "evaluations": [
            {
                "detailed_evaluations": {
                    "clarity": {
                        "score": 6.5,
                        "suggestions": ["表述需要更清晰", "逻辑需要更连贯"]
                    },
                    "completeness": {
                        "score": 7.5,
                        "suggestions": ["可以添加更多细节"]
                    }
                },
                "review_result": {
                    "suggestions": ["建议整体结构优化"]
                },
                "final_score": 7.0
            }
        ],
        "improvement_history": [
            {
                "version": 1,
                "changes": ["改进了表述", "添加了更多示例"],
                "feedback": "表述更清晰了，但还需要完善"
            }
        ],
        "references": [
            {
                "type": "similar_case",
                "content": "这是一个相关的参考案例"
            }
        ]
    }
    
    # 调用优化方法
    response = await optimizer_agent.optimize_proposal(**input_data)
    
    # 验证响应
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_proposer_agent_generate(proposer_agent):
    """测试提案生成器的generate方法"""
    # 准备测试数据
    input_data = {
        "input": "创建一个项目计划",
        "constraints": [
            {
                "type": "timeline",
                "value": "3个月内完成"
            },
            {
                "type": "budget",
                "value": "预算不超过100万"
            }
        ],
        "goals": [
            "提高系统性能",
            "优化用户体验"
        ],
        "references": [
            {
                "type": "case",
                "content": "类似项目案例：成功在2个月内完成了系统升级，性能提升30%"
            },
            {
                "type": "document",
                "content": "性能优化最佳实践指南",
                "metadata": {
                    "source": "技术文档库"
                }
            }
        ]
    }
    
    # 调用生成方法
    response = await proposer_agent.generate(**input_data)
    
    # 验证响应
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    
    # 测试无参考资料的情况
    input_data_no_refs = input_data.copy()
    input_data_no_refs.pop("references")
    response_no_refs = await proposer_agent.generate(**input_data_no_refs)
    assert response_no_refs is not None
    assert isinstance(response_no_refs, str)
    assert len(response_no_refs) > 0
    
    # 测试输入验证
    with pytest.raises(ValueError):
        # 测试无效的约束条件
        invalid_data = input_data.copy()
        invalid_data["constraints"] = [{"invalid": "constraint"}]
        await proposer_agent.generate(**invalid_data)
    
    with pytest.raises(ValueError):
        # 测试无效的目标格式
        invalid_data = input_data.copy()
        invalid_data["goals"] = "invalid goals"  # 应该是列表
        await proposer_agent.generate(**invalid_data)

def test_critic_agent_prompt_templates(critic_agent):
    assert isinstance(critic_agent.user_prompt_template, PromptTemplate)
    assert isinstance(critic_agent.system_prompt_template, PromptTemplate)

def test_optimizer_agent_prompt_templates(optimizer_agent):
    assert isinstance(optimizer_agent.system_prompt, PromptTemplate)
    assert isinstance(optimizer_agent.optimization_prompt, PromptTemplate)

def test_proposer_agent_prompt_templates(proposer_agent):
    assert isinstance(proposer_agent.system_prompt, PromptTemplate)
    assert isinstance(proposer_agent.base_prompt, PromptTemplate)
    assert isinstance(proposer_agent.rag_prompt, PromptTemplate)
