from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict, List, Any, Optional
from models.base import BaseQwenModel
from langsmith import traceable
from .prompts import CRITIC_PROMPTS
import logging
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenModel(BaseQwenModel):
    """Critic专用的deepseek-r1模型实现"""
    def __init__(self, model: str = "qwen-plus", api_version: str = "v1"):
        super().__init__(model=model, api_version=api_version, agent_name="critic")
        
    async def ainvoke(self, messages: List[Any], step_id: str) -> str:
        """调用Qwen模型，专门用于处理评估任务
        
        Args:
            messages: 消息列表
            step_id: 步骤ID
            
        Returns:
            str: 模型生成的文本
            
        Raises:
            ValueError: 如果模型调用失败或返回无效响应
        """
        return await self._base_ainvoke(
            messages=messages,
            step_id=step_id,
            extra_metadata={"task": "proposal_evaluation"}
        )


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
        self.model = QwenModel(model, api_version)
        self.output_parser = JsonOutputParser(pydantic_PLACEHOLDER_FOR_SECRET_ID)
        self.focus = focus or "default"
        
        if self.focus not in CRITIC_PROMPTS:
            raise ValueError(f"Unsupported focus: {focus}. Must be one of: {list(CRITIC_PROMPTS.keys())}")
        
        # 打印output_parser的格式说明
        format_instructions = self.output_parser.get_format_instructions()
        
        # 定义系统提示模板
        self.system_prompt_template = PromptTemplate(
            template=CRITIC_PROMPTS[self.focus],
            input_variables=[],
            partial_variables={"format_instructions": format_instructions}
        )
        
        # 定义用户提示模板
        self.user_prompt_template = PromptTemplate(
            template="""请评估以下提案：

问题：{input}

提案内容：
{proposal_content}

目标：
{goals_text}

约束条件：
{constraints_text}

请根据以上内容进行评估。特别关注提案的{focus}方面。
""",
            input_variables=["input", "proposal_content", "goals_text", "constraints_text", "focus"],
            partial_variables={"format_instructions": format_instructions}
        )

    @traceable(name="evaluate_proposal", run_type="chain")
    async def evaluate_proposal(
        self, 
        input: str,
        proposal_content: str,
        goals: List[str],
        constraints: List[Dict[str, str]],
        step_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """评估提案
        
        Args:
            input: 输入问题
            proposal_content: 提案内容字符串
            goals: 目标列表
            constraints: 约束条件列表
            step_id: 可选的步骤ID
            
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
            result = await self.model.ainvoke(
                messages=[system_msg, user_msg],
                step_id=step_id or "evaluate_proposal"
            )
            
            try:
                # 尝试解析评估结果
                evaluation = self.output_parser.parse(result)

                return {
                    "score": evaluation.get('overall_score', 0) / 10.0,  # 转换为0-1范围
                    "comments": evaluation.get('overall_feedback', ''),
                    "suggestions": [
                        f"{goal}: {eval_result.get('suggestions', '')}"
                        for goal, eval_result in evaluation.get('evaluations', {}).items()
                    ]
                }
                
            except Exception as e:
                logger.error(f"解析评估结果时出错: {e}")
                raise
                
        except Exception as e:
            logger.error(f"评估提案时出错: {e}")
            raise
