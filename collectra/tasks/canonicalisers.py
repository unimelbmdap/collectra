"""Matching text to the canonical form of an entity, with BM25 and an LLM.

Classes:
    LLMCanonicaliser: finds the entities closest to the text, then asks the LLM
        which one the text refers to.
"""

__all__ = ["LLMCanonicaliser"]

import re
from pathlib import Path

from dotenv import load_dotenv

from collectra.tasks.bm25_index import BM25Index
from collectra.tasks.llms import LLM as CollectraLLM
from collectra.types.texts import Text

load_dotenv()


class LLMCanonicaliser(CollectraLLM):
    """Asks the LLM which known entity the input text refers to.

    With an entity file, the ``count`` entities that best match the text (by
    BM25 over words and their character trigrams, so misspellings still match)
    are offered to the LLM. A list of entities is offered whole.

    Attributes:
        entities (list[str]|Path): The entities, or a file with one per line.
        count (int): How many entities from a file to offer.
    """

    def __init__(self, name: str, model: str, **kwargs):
        super().__init__(name, model, **kwargs)
        self.threshold = kwargs.get("threshold", 0.8)
        self.count = kwargs.get("count", 5)
        self.entities = kwargs.get("entities", [])
        entities_path = Path(self.entities)
        if not entities_path.exists() and entities_path.suffix == "":
            if isinstance(self.entities, str) or not isinstance(self.entities, list):
                self.entities = [entity for entity in str(self.entities).split(",")]
        else:
            self.entities = entities_path
        self._index: BM25Index | None = None

    def _entity_index(self) -> BM25Index:
        """The entity file's index, built the first time it is needed."""
        if self._index is None:
            with open(self.entities, "r") as f:
                names = [line.strip() for line in f if line.strip()]
            self._index = BM25Index([str(i) for i in range(len(names))], names)
        return self._index

    def retrieve_entities(self, query_data: str) -> dict[str, list[str]]:
        """The best-matching entities, best first, each with its line numbers."""
        entities: dict[str, list[str]] = {}
        for document, _ in self._entity_index().search(query_data, k=self.count):
            entities.setdefault(document.page_content, []).append(document.id)
        return entities

    def run(self, *args: Text) -> Text:
        """Canonicalize the input text against known entities.

        Args:
            text (str): Input text to canonicalize.

        Returns:
            str: Canonicalized text if a match is found, otherwise original text.
        """
        prompt, pattern = self.pre_run()

        entity_stem = re.sub("_.*", "", self.name)
        task_inputs = list(args)
        query_data: str = ", ".join(
            [
                str(task_input.data)
                for task_input in task_inputs
                if entity_stem in task_input.name
            ]
        )
        entities = ""

        if isinstance(self.entities, Path):
            entities_dict = self.retrieve_entities(query_data)
            entities = ", ".join(list(entities_dict.keys()))
        else:
            entities = ", ".join(self.entities)

        messages = self.replace_inputs(pattern, prompt, *args, entities=entities)

        self.messages.append(messages)

        self.check_inputs(*args)

        response = self.invoke()

        return self.text_output(response)
