from typing import Dict, List, Any, Optional, Union
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
    ANALYSIS_SYSTEM_PROMPT,
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

        # 定义分析系统提示模板
        self.anal_system_prompt = PromptTemplate(
            template=ANALYSIS_SYSTEM_PROMPT,
            input_variables=[],
        )

        # 分析提示模板
        self.analysis_prompt = PromptTemplate(
            template=OPTIMIZER_ANALYSIS_PROMPT,
            input_variables=["evaluation_results"]
        )

        # 优化提示模板
        self.optimization_prompt = PromptTemplate(
            template=OPTIMIZER_OPTIMIZATION_PROMPT,
            input_variables=["current_proposal", "analysis_result", "references"]
        )

        # 创建结构化输出的分析模型
        self.structured_analyzer = self.model.with_structured_output(AnalysisResult)

    def _format_evaluations(self, evaluations: List[Dict[str, Any]]) -> str:
        """格式化评估结果
        
        Args:
            evaluations: 评估结果列表
            
        Returns:
            格式化后的评估结果字符串
        """
        if not evaluations:
            return "暂无评估结果"
        
        # 只使用最新的评估结果
        latest_eval = evaluations[-1]
        return json.dumps(latest_eval, ensure_ascii=False, indent=2)

    def _format_analysis_result(self, analysis: Union[AnalysisResult, str]) -> str:
        """格式化分析结果
        
        Args:
            analysis: 分析结果，可能是 AnalysisResult 对象或字符串
            
        Returns:
            格式化后的分析结果字符串
        """
        if isinstance(analysis, str):
            return analysis
            
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

    def _format_references(self, references: Optional[List[Dict[str, Any]]]) -> str:
        """格式化参考资料
        
        Args:
            references: 参考资料列表
            
        Returns:
            格式化后的参考资料字符串
        """
        if not references:
            return "暂无参考资料"
            
        result = []
        for i, ref in enumerate(references, 1):
            result.append(f"## 参考资料 {i}")
            for key, value in ref.items():
                result.append(f"- {key}: {value}")
            result.append("")  # 空行分隔
            
        return "\n".join(result)

    async def optimize_proposal(
        self,
        current_proposal: str,
        evaluations: List[Dict[str, Any]],
        references: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """优化提案
        
        Args:
            current_proposal: 当前提案
            evaluations: 评估结果列表
            references: 可选的参考资料
            
        Returns:
            优化后的提案
        """
        try:
            # 第一轮：分析评估结果
            analysis_messages = [
                SystemMessage(content=self.anal_system_prompt.format()),
                HumanMessage(content=self.analysis_prompt.format(
                    evaluation_results=self._format_evaluations(evaluations)
                ))
            ]
            
            # 执行分析
            try:
                # 使用结构化输出直接获取AnalysisResult
                analysis_result = await self.structured_analyzer.ainvoke(input=analysis_messages)
            except Exception as e:
                # 如果结构化输出失败，回退到普通输出
                analysis_result = await self.model.ainvoke(input=analysis_messages)
                analysis_result = analysis_result.content
            
            # 第二轮：根据分析结果优化提案
            optimization_messages = [
                SystemMessage(content=self.system_prompt.format()),
                HumanMessage(content=self.optimization_prompt.format(
                    current_proposal=current_proposal,
                    analysis_result=self._format_analysis_result(analysis_result),
                    references=self._format_references(references)
                ))
            ]
            
            # 执行优化
            optimization_result = await self.model.ainvoke(input=optimization_messages)
            
            return optimization_result.content
            
        except Exception as e:
            logger.error(f"优化提案失败: {str(e)}")
            raise
