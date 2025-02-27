"""Define the configurable parameters for the proposal workflow."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional, Dict, Any

from langchain_core.runnables import RunnableConfig, ensure_config


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the proposal workflow."""

    # 主模型配置
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="qwen-max",
        metadata={
            "description": "The name of the language model to use for the main interactions. "
            "Should be in the form of a model name supported by the system."
        },
    )

    # 提案生成器模型配置
    proposer_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="qwen-max",
        metadata={
            "description": "The name of the language model to use for the proposal generator. "
            "If not specified, will use the main model."
        },
    )

    # 评估器模型配置
    critic_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="qwen-plus",
        metadata={
            "description": "The name of the language model to use for the critic agent. "
            "If not specified, will use the main model."
        },
    )

    # 优化器模型配置
    optimizer_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="qwen-max",
        metadata={
            "description": "The name of the language model to use for the optimizer agent. "
            "If not specified, will use the main model."
        },
    )

    # 最大迭代次数
    max_iterations: int = field(
        default=3,
        metadata={
            "description": "The maximum number of iterations to run the proposal workflow."
        },
    )

    # 优秀评分标准
    excellent_score: float = field(
        default=8.5,
        metadata={
            "description": "The score threshold for considering a proposal excellent. "
            "If the score is equal to or above this threshold, the workflow will stop."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
