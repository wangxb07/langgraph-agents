from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langsmith import traceable
import logging
import json
import numpy as np
from collections import defaultdict
from proposer.utils import init_custom_chat_model
from .prompts import (
    OPTIMIZER_SYSTEM_PROMPT, 
    OPTIMIZER_ANALYSIS_PROMPT,
    OPTIMIZER_OPTIMIZATION_PROMPT
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisResult(BaseModel):
    """分析结果模型"""
    key_issues: List[str] = Field(description="需要改进的关键问题")
    suggestions: Dict[str, List[str]] = Field(description="各维度的改进建议")

class OptimizerAgent:
    """提案优化器
    
    基于评估历史和改进建议优化提案内容。使用两轮对话实现：
    1. 第一轮：分析评估结果和历史记录，生成关键问题和建议
    2. 第二轮：基于分析结果优化提案
    """

    def __init__(self, model: str = "qwen-max", api_version: str = "v1"):
        """初始化提案优化器"""
        self.model = init_custom_chat_model(model)
        
        # 定义系统提示模板
        self.system_prompt = PromptTemplate(
            template=OPTIMIZER_SYSTEM_PROMPT,
            input_variables=[],
        )

        # 分析提示模板
        self.analysis_prompt = PromptTemplate(
            template=OPTIMIZER_ANALYSIS_PROMPT,
            input_variables=["current_proposal", "evaluation_results", "improvement_history"]
        )

        # 优化提示模板
        self.optimization_prompt = PromptTemplate(
            template=OPTIMIZER_OPTIMIZATION_PROMPT,
            input_variables=["current_proposal", "analysis_result", "references"]
        )

        # 创建结构化输出的分析模型
        self.structured_analyzer = self.model.with_structured_output(AnalysisResult)

    def _format_evaluations(self, evaluations: List[Dict[str, Any]]) -> str:
        """将评估结果格式化为易读的文档格式
        
        Args:
            evaluations: 评估结果列表
            
        Returns:
            str: 格式化后的评估文档
        """
        result = []
        for i, eval in enumerate(evaluations, 1):
            result.append(f"## 评估 {i}\n")
            
            # 处理详细评估
            if "detailed_evaluations" in eval:
                result.append("### 详细评估")
                for focus, detail in eval["detailed_evaluations"].items():
                    result.append(f"\n#### {focus}")
                    result.append(f"- 得分: {detail['score']}")
                    if "suggestions" in detail and detail["suggestions"]:
                        result.append("- 建议:")
                        for suggestion in detail["suggestions"]:
                            result.append(f"  * {suggestion}")
            
            # 处理评审结果
            if "review_result" in eval:
                result.append("\n### 评审结果")
                review = eval["review_result"]
                if "suggestions" in review:
                    result.append("建议:")
                    for suggestion in review["suggestions"]:
                        result.append(f"- {suggestion}")
            
            # 处理最终得分
            if "final_score" in eval:
                result.append(f"\n### 最终得分: {eval['final_score']}")
            
            result.append("\n" + "-" * 50 + "\n")  # 分隔线
        
        return "\n".join(result)

    def _format_improvement_history(self, history: List[Dict[str, Any]]) -> str:
        """将改进历史格式化为易读的文档格式
        
        Args:
            history: 改进历史记录列表
            
        Returns:
            str: 格式化后的历史文档
        """
        if not history:
            return "暂无改进历史"
            
        result = []
        for i, record in enumerate(history, 1):
            result.append(f"## 版本 {record.get('version', i)}\n")
            
            if "changes" in record:
                result.append("### 改动:")
                for change in record["changes"]:
                    result.append(f"- {change}")
            
            if "feedback" in record:
                result.append(f"\n### 反馈:")
                result.append(record["feedback"])
            
            result.append("\n" + "-" * 50 + "\n")  # 分隔线
        
        return "\n".join(result)

    def _format_analysis_result(self, analysis: AnalysisResult) -> str:
        """将分析结果格式化为易读的文档格式
        
        Args:
            analysis: 分析结果对象
            
        Returns:
            str: 格式化后的分析文档
        """
        result = []
        
        # 关键问题
        result.append("## 关键问题")
        for issue in analysis.key_issues:
            result.append(f"- {issue}")
        
        # 改进建议
        result.append("\n## 改进建议")
        for dimension, suggestions in analysis.suggestions.items():
            result.append(f"\n### {dimension}")
            for suggestion in suggestions:
                result.append(f"- {suggestion}")
        
        return "\n".join(result)

    async def optimize_proposal(self, 
                              current_proposal: str,
                              evaluations: List[Dict[str, Any]],
                              improvement_history: Optional[List[Dict[str, Any]]] = None,
                              references: Optional[List[Dict[str, Any]]] = None) -> str:
        """优化提案
        
        使用两轮对话进行优化：
        1. 第一轮：分析评估结果和历史，生成关键问题和建议
        2. 第二轮：基于分析结果优化提案
        
        Args:
            current_proposal: 当前提案内容
            evaluations: 评估结果列表
            improvement_history: 历史改进记录
            references: 参考资料
            
        Returns:
            str: 优化后的提案内容
        """
        try:
            # 第一轮：分析评估结果和历史
            analysis_messages = [
                SystemMessage(content=self.system_prompt.format()),
                HumanMessage(content=self.analysis_prompt.format(
                    current_proposal=current_proposal,
                    evaluation_results=self._format_evaluations(evaluations),
                    improvement_history=self._format_improvement_history(improvement_history or [])
                ))
            ]
            
            try:
                # 使用结构化输出直接获取AnalysisResult
                analysis_result = await self.structured_analyzer.ainvoke(input=analysis_messages)
            except Exception as e:
                logger.error(f"结构化分析失败: {str(e)}")
                # 如果结构化分析失败，使用默认值
                analysis_result = AnalysisResult(
                    key_issues=["分析失败，使用默认结果"],
                    suggestions={"general": ["请检查提案的整体质量"]}
                )
            
            # 第二轮：基于分析结果优化提案
            optimization_messages = [
                SystemMessage(content=self.system_prompt.format()),
                HumanMessage(content=self.optimization_prompt.format(
                    current_proposal=current_proposal,
                    analysis_result=self._format_analysis_result(analysis_result),
                    references=json.dumps(references or [], ensure_ascii=False, indent=2)
                ))
            ]
            
            optimized_proposal = await self.model.ainvoke(input=optimization_messages)
            return optimized_proposal.content
            
        except Exception as e:
            logger.error(f"提案优化失败: {str(e)}")
            raise
