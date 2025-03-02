from typing import Dict, List, Any, Optional, Tuple, Union, Sequence
from typing_extensions import Annotated
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from dataclasses import dataclass, field
import logging
import uuid
from proposer.agents.proposer.core import ProposerAgent
from proposer.agents.critic.core import CriticAgent
from proposer.agents.optimizer.core import OptimizerAgent
from proposer.configuration import Configuration

logger = logging.getLogger(__name__)

@dataclass
class ProposalInput:
    """提案生成的输入参数"""
    input: str
    constraints: List[Dict[str, str]]
    goals: List[str]


@dataclass
class ProposalState(ProposalInput):
    """提案工作流状态"""
    proposals: List[str] = field(default_factory=list)  # 存储所有版本的提案
    evaluations: List[Dict[str, Any]] = field(default_factory=list)
    status: str = field(default="init")
    iteration: int = field(default=0)
    max_iterations: int = field(default=3)
    excellent_score: float = field(default=8.5)

    @property
    def current_proposal(self) -> Optional[str]:
        """获取当前最新的提案"""
        return self.proposals[-1] if self.proposals else None

    @property
    def latest_evaluation(self) -> Optional[Dict[str, Any]]:
        """获取最新的评估结果"""
        return self.evaluations[-1] if self.evaluations else None


class ProposalWorkflow:
    """提案工作流
    
    实现提案的生成、评估和优化的完整工作流程。
    工作流程：
    1. 生成初始提案
    2. 评估提案
    3. 仲裁（决定是继续优化还是结束）
    4. 如果需要优化，返回步骤1进行新一轮迭代
    """
    
    def __init__(self):
        """初始化工作流"""
        self.proposer = None
        self.critics = {}  # 存储不同focus的评估代理
        self.optimizer = None
        self.configuration = None
    
    async def _init_agents(self, state: ProposalState, config: RunnableConfig) -> Dict[str, Any]:
        """初始化代理
        
        Args:
            state: 当前状态
            config: 运行时配置
            
        Returns:
            更新后的状态
        """
        try:
            # 从配置中获取参数
            self.configuration = Configuration.from_runnable_config(config)
            
            # 初始化代理
            self.proposer = ProposerAgent(model=self.configuration.proposer_model)
            
            # 初始化多个评估代理，每个关注不同的方面
            critic_model = self.configuration.critic_model
            self.critics = {
                "logic": CriticAgent(model=critic_model, focus="logic"),
                "completeness": CriticAgent(model=critic_model, focus="completeness"),
                "feasibility": CriticAgent(model=critic_model, focus="feasibility")
            }
            
            self.optimizer = OptimizerAgent(model=self.configuration.optimizer_model)
            
            # 更新状态中的配置参数
            state.max_iterations = self.configuration.max_iterations
            state.excellent_score = self.configuration.excellent_score
            
            return state.__dict__
        
        except Exception as e:
            logger.error(f"初始化代理失败: {str(e)}")
            raise
    
    async def _wait_user_feedback(self, state: ProposalState) -> Dict[str, Any]:
        """等待用户对评估结果的反馈
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        # 获取最新的提案和评估结果
        latest_proposal = state.current_proposal
        latest_evaluation = state.latest_evaluation
        
        # 构造要展示给用户的上下文
        context = {
            "evaluation": latest_evaluation,
        }
        
        # 获取用户对评估结果的修改
        evaluation_feedback = interrupt(
            {
                "type": "evaluation_feedback",
                "context": context,
                "message": "请查看并修改评估结果"
            }
        )
        
        # 更新评估结果
        if isinstance(evaluation_feedback, dict) and state.evaluations:
            state.evaluations[-1].update(evaluation_feedback)
        
        # 获取用户的继续/停止决定
        action = interrupt(
            {
                "type": "action_choice",
                "context": context,
                "message": "是否继续优化？",
                "options": ["continue", "stop"]
            }
        )
        
        # 更新状态
        if action == "stop":
            state.status = "completed"
        else:
            state.status = "refine"
        
        return state.__dict__

    async def _generate_proposal(self, state: ProposalState) -> Dict[str, Any]:
        """生成或优化提案
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            current_state = state
            
            # 自动从 RAG 中检索相关文档
            from proposer.tools import get_rag_tool
            rag_tool = get_rag_tool()
            references = []
            retrieved_docs = rag_tool.retrieve(current_state.input)
            
            for doc in retrieved_docs:
                references.append({
                    "type": "document",
                    "content": doc["content"],
                    "metadata": doc["metadata"]
                })
            
            if not current_state.proposals:
                # 首次生成提案
                proposal = await self.proposer.generate(
                    input=current_state.input,
                    constraints=current_state.constraints,
                    goals=current_state.goals,
                    PLACEHOLDER_FOR_SECRET_ID  # 传入检索到的参考资料
                )
            else:
                # 优化现有提案
                proposal = await self.optimizer.optimize_proposal(
                    current_proposal=current_state.current_proposal,
                    evaluations=current_state.evaluations,
                    PLACEHOLDER_FOR_SECRET_ID  # 传入检索到的参考资料
                )
            
            # 更新状态
            current_state.iteration += 1
            current_state.proposals.append(proposal)
            current_state.status = "generated"
            
            return current_state.__dict__
        
        except Exception as e:
            logger.error(f"生成提案失败: {str(e)}")
            raise
    
    async def _evaluate_proposal(self, state: ProposalState) -> Dict[str, Any]:
        """评估提案
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            current_state = state
            
            # 使用多个评估代理对提案进行多维度评估
            evaluation_results = {}
            
            # 对每个维度进行评估
            for focus, critic in self.critics.items():
                result = await critic.evaluate_proposal(
                    input=current_state.input,
                    proposal_content=current_state.current_proposal,
                    goals=current_state.goals,
                    constraints=current_state.constraints
                )
                evaluation_results[focus] = result
            
            # 计算综合评分（各维度的平均值）
            overall_score = sum(result["score"] for result in evaluation_results.values()) / len(evaluation_results)
            
            # 整合评估结果
            combined_evaluation = {
                "score": overall_score,
                "dimensions": evaluation_results,
                "comments": "多维度评估结果汇总：\n" + "\n".join([
                    f"- {focus}：{result['comments']}" 
                    for focus, result in evaluation_results.items()
                ]),
                "suggestions": []
            }
            
            # 整合所有维度的建议
            for focus, result in evaluation_results.items():
                combined_evaluation["suggestions"].extend([
                    f"[{focus}] {suggestion}" for suggestion in result.get("suggestions", [])
                ])
            
            # 更新状态
            current_state.evaluations.append(combined_evaluation)
            current_state.status = "evaluated"
            
            return current_state.__dict__
            
        except Exception as e:
            logger.error(f"评估提案失败: {str(e)}")
            raise
    
    async def _arbitrate(self, state: ProposalState) -> Union[str, Tuple[str, ProposalState]]:
        """仲裁决定是否需要继续优化
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            # 获取最新的评估结果
            latest_evaluation = state.latest_evaluation
            
            # 判断是否需要继续优化
            if state.iteration >= state.max_iterations:
                # 达到最大迭代次数
                state.status = "completed"
            elif latest_evaluation["score"] >= state.excellent_score:
                # 评分达到优秀标准
                state.status = "completed"
            else:
                # 需要继续优化
                state.status = "refine"
            
            return state.__dict__
            
        except Exception as e:
            logger.error(f"仲裁失败: {str(e)}")
            raise
    
    def create_graph(self) -> Any:
        """创建工作流图
        
        Returns:
            编译后的工作流图
        """
        workflow = StateGraph(ProposalState, input=ProposalInput, config_PLACEHOLDER_FOR_SECRET_ID)
        
        # 添加节点
        workflow.add_node("init_agents", self._init_agents)
        workflow.add_node("propose", self._generate_proposal)
        workflow.add_node("evaluate", self._evaluate_proposal)
        workflow.add_node("arbitrate", self._arbitrate)
        
        # 设置工作流程
        workflow.set_entry_point("init_agents")
        
        # 添加边
        workflow.add_edge("init_agents", "propose")
        workflow.add_edge("propose", "evaluate")
        workflow.add_edge("evaluate", "arbitrate")
        
        # 添加条件边 - 根据用户反馈决定是继续优化还是结束
        workflow.add_conditional_edges(
            "arbitrate",
            lambda state: "propose" if state.status == "refine" else "end",
            {
                "propose": "propose",  # 继续优化
                "end": END  # 结束工作流
            }
        )
        
        # 设置工作流名称
        workflow.name = "Proposal Workflow"
        
        # 添加 checkpointer 支持 interrupt
        checkpointer = MemorySaver()
        workflow = workflow.compile(PLACEHOLDER_FOR_SECRET_ID)
        
        return workflow


def create_graph():
    """创建提案工作流图
    
    Returns:
        编译后的工作流图
    """
    workflow = ProposalWorkflow()
    return workflow.create_graph()
