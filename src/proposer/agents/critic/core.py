from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from langsmith import traceable
from .prompts import CRITIC_PROMPTS, CRITIC_USER_PROMPT_TEMPLATE
from proposer.utils import init_custom_chat_model
import logging
import json

logger = logging.getLogger(__name__)

class GoalEvaluation(BaseModel):
    """单个目标的评估结果"""
    score: float = Field(description="Score between 1-10")
    feedback: str = Field(description="Detailed feedback explaining the score")
    suggestions: str = Field(description="Improvement suggestions")

class EvaluationResponse(BaseModel):
    """完整的评估响应"""
    evaluations: Dict[str, GoalEvaluation] = Field(description="Evaluation results for each goal")
    overall_feedback: str = Field(description="Overall synthesis of the evaluation")
    overall_score: float = Field(description="Overall score between 1-10")

class CriticAgent:
    """评估代理
    
    支持多种评估取向的专业评估代理。可以根据不同的评估重点（如逻辑性、完整性等）
    提供针对性的评估。
    """
    
    def __init__(self, model: str = "qwen-plus", api_version: str = "v1", focus: Optional[str] = None):
        """初始化评估代理
        
        Args:
            model: 使用的模型名称
            api_version: API版本
            focus: 评估重点，可选值：logic, completeness, innovation, feasibility
                  如果为None，则使用默认的通用评估模板
        """
        self.model = init_custom_chat_model(model).with_structured_output(EvaluationResponse)
        self.focus = focus or "default"
        
        if self.focus not in CRITIC_PROMPTS:
            raise ValueError(f"Unsupported focus: {focus}. Must be one of: {list(CRITIC_PROMPTS.keys())}")
        
        # 定义系统提示模板
        self.system_prompt_template = PromptTemplate(
            template=CRITIC_PROMPTS[self.focus],
            input_variables=[]
        )
        
        # 定义用户提示模板
        self.user_prompt_template = PromptTemplate(
            template=CRITIC_USER_PROMPT_TEMPLATE,
            input_variables=["input", "proposal_content", "goals_text", "constraints_text", "focus"]
        )

    @traceable(name="evaluate_proposal", run_type="chain")
    async def evaluate_proposal(
        self, 
        input: str,
        proposal_content: str,
        goals: List[str],
        constraints: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """评估提案
        
        Args:
            input: 输入问题
            proposal_content: 提案内容
            goals: 目标列表
            constraints: 约束条件列表
            
        Returns:
            Dict[str, Any]: 评估结果，包含分数、评论和改进建议
        """
        try:
            # 准备系统消息
            system_msg = SystemMessage(content=self.system_prompt_template.format())
            
            # 准备用户消息
            user_msg = HumanMessage(content=self.user_prompt_template.format(
                input=input,
                proposal_content=proposal_content,
                goals_text="\n".join(f"- {goal}" for goal in goals),
                constraints_text="\n".join(f"- {c['type']}: {c['value']}" for c in constraints),
                focus=self.focus
            ))
            
            # 评估提案
            evaluation = await self.model.ainvoke(
                input=[system_msg, user_msg]
            )
            
            # 转换为标准输出格式
            return {
                "score": evaluation.overall_score / 10.0,  # 转换为0-1范围
                "comments": evaluation.overall_feedback,
                "suggestions": [
                    f"{goal}: {eval_result.suggestions}"
                    for goal, eval_result in evaluation.evaluations.items()
                ]
            }

        except Exception as e:
            logger.error(f"评估提案时出错: {e}")
            raise
