from pathlib import Path
from difflib import get_close_matches, SequenceMatcher
from collectra.tasks.base import Task
from dataclasses import dataclass


@dataclass(kw_only=True)
class EntityCanonicalizer(Task):
    """Canonicalizes entities in the text."""
    entities: list[str]|Path = None
    threshold: float = 0.8

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.entities, Path) and self.entities.is_file():
            with open(self.entities, "r") as f:
                self.entities = [line.strip() for line in f if line.strip()]

    def input_types(self) -> type:
        return str

    def output_types(self) -> type:
        return str
    
    def run(self, text: str) -> str:
        close_matches = get_close_matches(
            text,
            self.entities,
            cutoff=self.threshold,
            n=1,
        )

        match_score = 0
        if close_matches:
            match_score = round(SequenceMatcher(None, text, close_matches[0]).ratio(), 3)
            text = close_matches[0]
            
        # Do something with match_score
        # i.e. save to YAML output somehow
        return text

