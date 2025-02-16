from typing import Dict, List, Any, Optional
from models.base import BaseQwenModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable
import logging
import json
import numpy as np
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenModel(BaseQwenModel):
    """Optimizer专用的Qwen模型实现"""
    def __init__(self, model: str = "qwen-plus", api_version: str = "v1"):
        super().__init__(model=model, api_version=api_version, agent_name="optimizer")
        
    @traceable(run_type="prompt", name="optimizer")
    async def ainvoke(self, messages: List[Any], step_id: str) -> str:
        """调用Qwen模型，专门用于处理提案优化任务"""
        return await self._base_ainvoke(
            messages=messages,
            step_id=step_id,
            extra_metadata={"task": "proposal_optimization"},
            temperature=0.8  # 使用较低的temperature以保持一致性
        )

class OptimizerAgent:
    """提案优化器
    
    基于评估历史和改进建议优化提案内容。主要关注：
    1. 评估结果的分析和利用
    2. 历史改进记录的追踪
    3. 针对性的优化建议
    4. 持续迭代改进
    """

    def __init__(self, model: str = "qwen-max", api_version: str = "v1"):
        """初始化提案优化器"""
        self.model = QwenModel(model, api_version)
        
        # 定义系统提示模板
        self.system_prompt = PromptTemplate(
            template="""您是提案优化专家。基于评估历史和改进建议，生成一个优化后的提案版本。

评估重点：
1. 各部分评分和具体改进建议
2. 历史改进记录的参考
3. 整体提案质量提升

优化原则：
1. 保持提案核心价值
2. 针对性解决评估中的问题
3. 确保前后版本的连贯性
4. 突出改进点的实质性变化
""",
            input_variables=[]
        )

        # 优化提示模板
        self.optimization_prompt = PromptTemplate(
            template="""请基于以下信息优化提案：

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

生成一个优化后的提案版本。""",
            input_variables=["current_proposal", "evaluation_results", "improvement_history", "references"]
        )

    async def optimize_proposal(self, 
                              current_proposal: str,
                              evaluations: List[Dict[str, Any]],
                              improvement_history: Optional[List[Dict[str, Any]]] = None,
                              references: Optional[List[Dict[str, Any]]] = None) -> str:
        """优化提案
        
        Args:
            current_proposal: 当前提案内容
            evaluations: 评估结果列表，每个评估包含 detailed_evaluations, review_result, final_score
            improvement_history: 历史改进记录
            references: 参考资料，包括RAG检索结果和历史案例
            
        Returns:
            str: 优化后的提案内容
        """
        try:
            # 准备优化提示
            messages = [
                SystemMessage(content=self.system_prompt.format()),
                HumanMessage(content=self.optimization_prompt.format(
                    current_proposal=current_proposal,
                    evaluation_results=json.dumps(self._analyze_evaluations(evaluations), ensure_ascii=False, indent=2),
                    improvement_history=json.dumps(improvement_history or [], ensure_ascii=False, indent=2),
                    references=json.dumps(references or [], ensure_ascii=False, indent=2)
                ))
            ]
            
            # 生成优化后的提案
            optimized_proposal = await self.model.ainvoke(messages=messages, step_id="optimize_proposal")
            return optimized_proposal
            
        except Exception as e:
            logger.error(f"提案优化失败: {str(e)}")
            raise

    def _analyze_evaluations(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析评估结果，识别需要改进的重点领域
        
        Args:
            evaluations: 评估结果列表，每个评估包含 detailed_evaluations, review_result, final_score
            
        Returns:
            分析结果，包含：
            - 各维度评分统计
            - 主要问题点
            - 改进建议汇总
        """
        analysis = {
            'scores': defaultdict(list),
            'key_issues': [],  # 改用列表而不是集合
            'suggestions': defaultdict(list)
        }
        
        for eval in evaluations:
            # 处理详细评估
            for focus, detail in eval['detailed_evaluations'].items():
                analysis['scores'][focus].append(detail['score'])
                if detail['suggestions']:
                    analysis['suggestions'][focus].extend(detail['suggestions'])
                    
            # 处理评审结果
            if 'review_result' in eval:
                review = eval['review_result']
                if 'suggestions' in review:
                    analysis['suggestions']['review'].extend(review['suggestions'])
                    
        # 计算平均分并识别问题领域
        for focus, scores in analysis['scores'].items():
            avg_score = np.mean(scores)
            if avg_score < 7.0:  # 假设7分为及格线
                analysis['key_issues'].append(focus)  # 使用append而不是add
                
        # 将defaultdict转换为普通dict以便JSON序列化
        analysis['scores'] = dict(analysis['scores'])
        analysis['suggestions'] = dict(analysis['suggestions'])
                
        return analysis
