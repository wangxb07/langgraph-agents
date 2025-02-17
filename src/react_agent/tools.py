"""This module provides example tools for web scraping and search functionality.

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List

from react_agent.configuration import Configuration


TOOLS: List[Callable[..., Any]] = []
