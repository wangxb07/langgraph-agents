from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, List, Any, Optional
from models.base import BaseQwenModel
from langsmith import traceable
import logging
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO)

class QwenModel(BaseQwenModel):
    """Proposer专用的Qwen模型实现"""
    def __init__(self, model: str = "qwen-plus", api_version: str = "v1"):
        super().__init__(model=model, api_version=api_version, agent_name="proposer")

    async def ainvoke(self, messages: List[Any], step_id: str) -> str:
        """调用Qwen模型，专门用于处理提案生成任务
        
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
            extra_metadata={"task": "proposal_generation"},
            temperature=1.5
        )

class ProposerAgent:
    """提案生成器，负责生成提案内容"""
    
    def __init__(self, model: str = "qwen-max", api_version: str = "v1"):
        """初始化提案生成器
        
        Args:
            model: 模型名称
            api_version: API版本
        """
        self.model = QwenModel(model, api_version)
        
        # 定义系统提示模板
        self.system_prompt = PromptTemplate(
            template="""您是资深运营专家，请基于以下目标和约束生成创新方案。

目标：
{goals}

约束：
{constraints}

""",
            input_variables=["goals", "constraints"]
        )
        
        # 定义用户提示模板（基础版本）
        self.base_prompt = PromptTemplate(
            template="""基于以下背景，生成详细的创新方案：
{input}

请确保内容尽量详细。
""",
            input_variables=["input"]
        )
        
        # 定义用户提示模板（RAG增强版本）
        self.rag_prompt = PromptTemplate(
            template="""基于以下背景和参考资料，生成详细的创新方案：
{input}

# 参考资料
{references_text}

要求：
1. 充分利用参考资料中的相关信息
""",
            input_variables=["input", "references_text"]
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
            elif ref.get("type") == "improvement_history":
                # 处理改进历史
                formatted_refs.append(f"改进历史:\n{ref['content']}")
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
        if not isinstance(goals, list):
            raise ValueError("goals must be a list")
        
        if not isinstance(constraints, list):
            raise ValueError("constraints must be a list")
        
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
        references: Optional[List[Dict[str, Any]]] = None,
        step_id: Optional[str] = None
    ) -> str:
        """生成提案
        
        Args:
            input: 输入信息
            constraints: 约束条件列表
            goals: 目标列表
            references: 可选的参考资料列表
            step_id: 可选的步骤ID
            
        Returns:
            生成的提案文本
        """
        try:
            # 验证输入
            self._validate_input(input, constraints, goals)
            
            # 准备系统消息
            system_msg = SystemMessage(content=self.system_prompt.format(
                goals="\n".join(f"- {goal}" for goal in goals),
                constraints="\n".join(f"- {c['type']}: {c['value']}" for c in constraints)
            ))
            
            # 准备用户消息
            if references:
                # 使用RAG增强版本的提示
                references_text = self._format_references(references)
                user_msg = HumanMessage(content=self.rag_prompt.format(
                    input=input,
                    references_text=references_text
                ))
            else:
                # 使用基础版本的提示
                user_msg = HumanMessage(content=self.base_prompt.format(input=input))
            
            # 生成提案
            messages = [system_msg, user_msg]
            response = await self.model.ainvoke(messages, step_id=step_id or "generate_proposal")
            
            return response
            
        except Exception as e:
            logger.error(f"生成提案失败: {e}")
            raise
