"""Text handling and processing classes for Collectra workflows.

This module provides basic text data structures and utilities for handling
textual content within Collectra workflows. It supports text content with
optional path associations for file-based text processing tasks.

Classes:
    Text: Basic text container with content and optional path metadata
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(kw_only=True)
class Text:
    """Basic text container for handling textual content in workflows.

    Provides a simple data structure for storing text content along with
    optional path metadata for file-based text processing operations.

    Attributes:
        content (str): The actual text content.
        path (Path): Optional path to the source file.
    """

    content: str
    path: Path

    def __init__(self, content: str):
        """Initialize a Text instance with content.

        Args:
            content (str): The text content to store.
        """
        self.content = content

    def __str__(self):
        """Return the text content as a string.

        Returns:
            str: The stored text content.
        """
        return self.content
