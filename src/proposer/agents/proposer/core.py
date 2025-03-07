from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, List, Any, Optional
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from proposer.utils import init_custom_chat_model
from .prompts import PROPOSER_SYSTEM_PROMPT, PROPOSER_BASE_PROMPT, PROPOSER_RAG_PROMPT
import logging
from langsmith import traceable

logger = logging.getLogger(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO)

class ProposerAgent:
    """提案生成器，负责生成提案内容"""
    
    def __init__(self, model: str = "qwen-max", api_version: str = "v1"):
        """初始化提案生成器
        
        Args:
            model: 模型名称
            api_version: API版本
        """
        self.model = init_custom_chat_model(model)
        
        # 定义系统提示模板
        self.system_prompt = PromptTemplate(
            template=PROPOSER_SYSTEM_PROMPT,
            input_variables=[],
            partial_variables={}
        )
        
        # 定义用户提示模板（基础版本）
        self.base_prompt = PromptTemplate(
            template=PROPOSER_BASE_PROMPT,
            input_variables=["input", "goals_text", "constraints_text"]
        )
        
        # 定义用户提示模板（RAG增强版本）
        self.rag_prompt = PromptTemplate(
            template=PROPOSER_RAG_PROMPT,
            input_variables=["input", "goals_text", "constraints_text", "references_text"]
        )
        
    def _format_references(self, references: List[Dict[str, Any]]) -> str:
        """格式化参考资料
        
        Args:
            references: 参考资料列表
            
        Returns:
            格式化后的参考资料文本
        """
        formatted_refs = []
        for i, ref in enumerate(references):
            if ref.get("type") == "case":
                # 处理历史案例
                formatted_refs.append(f"历史案例 {i+1}:\n{ref['content']}")
            else:
                # 处理RAG检索的文档
                formatted_refs.append(
                    f"参考文档 {i+1}:\n{ref['content']}\n"
                    f"来源: {ref.get('metadata', {}).get('source', '未知')}"
                )
        
        return "\n\n".join(formatted_refs)
        
    def _validate_input(self, input: str, constraints: List[Dict], goals: List[str]):
        """验证输入参数
        
        Args:
            input: 输入信息
            constraints: 约束条件列表
            goals: 目标列表
            
        Raises:
            ValueError: 如果输入参数无效
        """        
        for constraint in constraints:
            if not isinstance(constraint, dict):
                raise ValueError("each constraint must be a dictionary")
            if "type" not in constraint or "value" not in constraint:
                raise ValueError("each constraint must have 'type' and 'value' fields")
    
    @traceable(name="generate_proposal", run_type="chain")
    async def generate(
        self,
        input: str,
        constraints: List[Dict],
        goals: List[str],
        references: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """生成提案
        
        Args:
            input: 输入信息
            constraints: 约束条件列表
            goals: 目标列表
            references: 可选的参考资料列表
            
        Returns:
            生成的提案文本
        """
        try:
            # 验证输入
            self._validate_input(input, constraints, goals)
            
            # 格式化目标和约束条件
            goals_text = "\n".join(f"- {goal}" for goal in goals)
            constraints_text = "\n".join(f"- {c['type']}: {c['value']}" for c in constraints)
            
            # 准备系统消息
            system_msg = SystemMessage(content=self.system_prompt.format())
            
            # 准备用户消息
            if references:
                # 使用RAG增强版本的提示
                references_text = self._format_references(references)
                user_msg = HumanMessage(content=self.rag_prompt.format(
                    input=input,
                    goals_text=goals_text,
                    constraints_text=constraints_text,
                    references_text=references_text
                ))
            else:
                # 使用基础版本的提示
                user_msg = HumanMessage(content=self.base_prompt.format(
                    input=input,
                    goals_text=goals_text,
                    constraints_text=constraints_text
                ))
            
            # 生成提案
            messages = [system_msg, user_msg]
            response = await self.model.ainvoke(input=messages)
            
            return response.content
            
        except Exception as e:
            logger.error(f"生成提案失败: {e}")
            raise
