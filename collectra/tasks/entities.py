"""Entity canonicalization and text processing tasks for Collectra workflows.

This module provides specialized tasks for entity recognition and canonicalization
operations. It includes functionality for matching text entities against known
canonical forms using fuzzy string matching algorithms.

The module supports:
    - Entity canonicalization with configurable similarity thresholds
    - Fuzzy string matching using difflib algorithms
    - Entity lists loaded from files or provided directly

Classes:
    EntityCanonicalizer: Task for canonicalizing entities in text
"""

from pathlib import Path
from difflib import get_close_matches, SequenceMatcher
from collectra.tasks.base import Task
from dataclasses import dataclass, field


@dataclass(kw_only=True)
class EntityCanonicalizer(Task):
    """Task for canonicalizing entities in text using fuzzy string matching.
    
    Matches input text against a list of known entities and returns the best
    canonical match if it exceeds the similarity threshold. Can load entity
    lists from files or use provided lists directly.
    
    Attributes:
        entities (list[str]|Path): List of canonical entities or path to file containing them.
        threshold (float): Minimum similarity threshold for matches (0.0 to 1.0).
    """
    entities: list[str]|Path = field(default_factory=list)
    threshold: float = 0.8

    def __post_init__(self):
        """Load entities from file if a Path is provided."""
        if isinstance(self.entities, Path) and self.entities.is_file():
            with open(self.entities, "r") as f:
                self.entities = [line.strip() for line in f if line.strip()]

    def input_type(self) -> type:
        """Define input type as string text.
        
        Returns:
            type: String type for text input.
        """
        return str

    def output_type(self) -> type:
        """Define output type as string text.
        
        Returns:
            type: String type for canonicalized text output.
        """
        return str
    
    def run(self, text: str) -> str:
        """Canonicalize the input text against known entities.
        
        Args:
            text (str): Input text to canonicalize.
            
        Returns:
            str: Canonicalized text if a match is found, otherwise original text.
        """
        if not isinstance(self.entities, list):
            return text
        close_matches = get_close_matches(text, self.entities, n=3, cutoff=self.threshold)
        match_score = 0
        if close_matches:
            match_score = round(SequenceMatcher(None, text, close_matches[0]).ratio(), 3)
            text = close_matches[0]
            
        # Do something with match_score
        # i.e. save to YAML output somehow
        return text

