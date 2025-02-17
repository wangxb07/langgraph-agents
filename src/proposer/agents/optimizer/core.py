from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langsmith import traceable
import logging
import json
import numpy as np
from collections import defaultdict
from proposer.utils import init_custom_chat_model
from .prompts import OPTIMIZER_SYSTEM_PROMPT, OPTIMIZER_OPTIMIZATION_PROMPT

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.model = init_custom_chat_model(model)
        
        # 定义系统提示模板
        self.system_prompt = PromptTemplate(
            template=OPTIMIZER_SYSTEM_PROMPT,
            input_variables=[],
            partial_variables={}
        )

        # 优化提示模板
        self.optimization_prompt = PromptTemplate(
            template=OPTIMIZER_OPTIMIZATION_PROMPT,
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
