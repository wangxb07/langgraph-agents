from typing import Dict, List, Any, Optional, Tuple, Union, Sequence
from typing_extensions import Annotated
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from dataclasses import dataclass, field
import logging
import uuid
from proposer.agents.proposer.core import ProposerAgent
from proposer.agents.critic.core import CriticAgent
from proposer.agents.optimizer.core import OptimizerAgent

logger = logging.getLogger(__name__)

@dataclass
class ProposalInput:
    """提案生成的输入参数"""
    input: str
    constraints: List[Dict[str, str]]
    goals: List[str]
    references: Optional[List[Dict[str, Any]]] = field(default=None)


@dataclass
class ProposalState(ProposalInput):
    """提案工作流状态"""
    proposals: List[str] = field(default_factory=list)  # 存储所有版本的提案
    evaluations: List[Dict[str, Any]] = field(default_factory=list)
    status: str = field(default="init")
    iteration: int = field(default=0)
    max_iterations: int = field(default=3)

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
    
    def __init__(self, 
                 model: str = "qwen-max",
                 max_iterations: int = 3):
        """初始化工作流
        
        Args:
            model: 使用的模型名称
            max_iterations: 最大迭代次数
        """
        self.proposer = ProposerAgent(model=model)
        self.critic = CriticAgent(model=model)
        self.optimizer = OptimizerAgent(model=model)
        self.max_iterations = max_iterations
    
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
            
            if not current_state.proposals:
                # 首次生成提案
                proposal = await self.proposer.generate(
                    input=current_state.input,
                    constraints=current_state.constraints,
                    goals=current_state.goals,
                    references=current_state.references
                )
            else:
                # 优化现有提案
                proposal = await self.optimizer.optimize_proposal(
                    current_proposal=current_state.current_proposal,
                    evaluations=current_state.evaluations,
                    references=current_state.references
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
            
            # 对提案进行评估
            evaluation = await self.critic.evaluate_proposal(
                input=current_state.input,
                proposal_content=current_state.current_proposal,  # 使用最新的提案
                goals=current_state.goals,
                constraints=current_state.constraints
            )
            
            # 更新状态
            current_state.evaluations.append(evaluation)
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
            elif latest_evaluation["score"] >= 8.5:
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
        workflow = StateGraph(ProposalState, input=ProposalInput)
        
        # 添加节点
        workflow.add_node("propose", self._generate_proposal)
        workflow.add_node("evaluate", self._evaluate_proposal)
        workflow.add_node("arbitrate", self._arbitrate)
        
        # 设置工作流程
        workflow.set_entry_point("propose")
        
        # 添加边
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

    async def run(self, 
                  input: str,
                  constraints: List[Dict[str, str]],
                  goals: List[str],
                  references: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """运行工作流
        
        Args:
            input: 输入信息
            constraints: 约束条件列表
            goals: 目标列表
            references: 可选的参考资料
            
        Returns:
            工作流执行结果
        """
        # 准备初始状态
        initial_state = ProposalState(
            input=input,
            PLACEHOLDER_FOR_SECRET_ID,
            goals=goals,
            PLACEHOLDER_FOR_SECRET_ID,
            max_iterations=self.max_iterations
        )
        
        # 创建并运行工作流
        graph = self.create_graph()
        
        # 配置线程 ID
        thread_id = str(uuid.uuid4())
        thread_config = {"configurable": {"thread_id": thread_id}}
        
        # 运行工作流
        final_state = await graph.arun(initial_state.__dict__, config=thread_config)
        
        return final_state

def create_graph():
    """创建提案工作流图
    
    Returns:
        编译后的工作流图
    """
    workflow = ProposalWorkflow(model="qwen-max", max_iterations=3)
    return workflow.create_graph()
