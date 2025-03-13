"""This module provides example tools for arithmetic operations.

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List

from langchain_core.tools import tool

from tongyichat_agent.configuration import Configuration


@tool("add", description="将两个数字相加")
def add(a: float, b: float) -> float:
    """将两个数字相加。

    Args:
        a (float): 第一个数字
        b (float): 第二个数字

    Returns:
        float: 两个数字的和
    """
    return a + b


@tool("subtract", description="将两个数字相减")
def subtract(a: float, b: float) -> float:
    """将两个数字相减。

    Args:
        a (float): 被减数
        b (float): 减数

    Returns:
        float: 两个数字的差
    """
    return a - b


@tool("multiply", description="将两个数字相乘")
def multiply(a: float, b: float) -> float:
    """将两个数字相乘。

    Args:
        a (float): 第一个数字
        b (float): 第二个数字

    Returns:
        float: 两个数字的积
    """
    return a * b


@tool("divide", description="将两个数字相除")
def divide(a: float, b: float) -> float:
    """将两个数字相除。

    Args:
        a (float): 被除数
        b (float): 除数

    Returns:
        float: 两个数字的商
        
    Raises:
        ValueError: 当除数为0时抛出异常
    """
    if b == 0:
        raise ValueError("除数不能为0")
    return a / b


TOOLS: List[Callable[..., Any]] = [
    add,
    subtract,
    multiply,
    divide,
]
