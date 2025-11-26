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

__all__ = ["LLMCanonicalizer"]

import re, yaml, re

from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from collectra.types.texts import Text
from collectra.tasks.llms import LLM as CollectraLLM

load_dotenv()

class LLMCanonicalizer(CollectraLLM):
    """Task for canonicalizing entities in text using fuzzy string matching.

    Matches input text against a list of known entities and returns the best
    canonical match if it exceeds the similarity threshold. Can load entity
    lists from files or use provided lists directly.

    Attributes:
        entities (list[str]|Path): List of canonical entities or path to file containing them.
        threshold (float): Minimum similarity threshold for matches (0.0 to 1.0).
    """

    def __init__(self, name: str, model: str, **kwargs):
        super().__init__(name, model, **kwargs)
        self.embedding_model = kwargs.get("embedding_model", "text-embedding-3-large")        
        self.threshold = kwargs.get("threshold", 0.8)
        self.count = kwargs.get("count", 5)
        self.entities = kwargs.get("entities", [])
        entities_path = Path(self.entities)
        if not entities_path.exists() and entities_path.suffix == "":
            if isinstance(self.entities, str) or not isinstance(self.entities, list):
                self.entities = [entity for entity in str(self.entities).split(",")]
        else:
            self.entities = entities_path

    def retrieve_entities(self, query_data: str) -> dict:                        
        index_path = Path(self.entities.stem + "_index")        
        embeddings = AzureOpenAIEmbeddings(model = self.embedding_model)
        if index_path.exists() and index_path.is_dir():                        
            print(f"Loading existing vector store from {index_path}")
            vector_store = Chroma(
                collection_name=f"{self.entities.stem}_collection",
                embedding_function=embeddings,
                persist_directory=str(index_path),
            )                           
        else:                        
            with open(self.entities, 'r') as f:
                raw_data = yaml.safe_load(f)
                ids = [str(id) for id in list(raw_data.keys())]
                documents = [Document(page_content=content) for content in raw_data.values()]
            vector_store = Chroma(
                collection_name=f"{self.entities.stem}_collection",
                embedding_function=embeddings,
                persist_directory=str(index_path),
            )   
            vector_store.add_documents(documents=documents, ids=ids)                                        

        entities = dict()

        results = vector_store.similarity_search_with_score(query_data, k=self.count)
        for result in results:
            if result[0].page_content in entities:
                entities[result[0].page_content].append(result[0].id)
            else:
                entities[result[0].page_content] = [result[0].id]              
        return entities      

    def run(self, *args: Text) -> Text:
        """Canonicalize the input text against known entities.

        Args:
            text (str): Input text to canonicalize.

        Returns:
            str: Canonicalized text if a match is found, otherwise original text.
        """                        
        entity_stem = re.sub("_.*", "", self.name)
        task_inputs = list(args)
        query_data: str | list = list()
        for task_input in task_inputs:
            if entity_stem in task_input.name:
                query_data.append(task_input.data)
        query_data = ", ".join(query_data)

        entities = ""
    
        if isinstance(self.entities, Path):
            entities_dict = self.retrieve_entities(query_data)                
            entities = ", ".join(list(entities_dict.keys()))
        elif isinstance(self.entities, list):
            entities = ", ".join(self.entities)
        elif isinstance(self.entities, str):
            entities = self.entities        

        prompt, pattern, _ = self.get_pattern_matches()
        messages: list[str | dict] = list()

        while re.search(pattern, prompt):
            match = next(re.finditer(pattern, prompt))
            start, end = match.span()
            item = match[1].strip()
            replaced = False
            if prompt[:start]:
                messages.append(self._add_text(prompt[:start]))
            for arg in args:                
                key = arg.name
                if key == item:
                    messages.append(self._add_content(arg))
                    replaced = True
                    break
            if item == "entities":
                replaced = True
                messages.append(self._add_text(entities))
            if not replaced:
                messages.append(self._add_text(f"No content provided for {item}. Ignore this part."))
            prompt = prompt[end:].strip()
        
        if prompt:
            messages.append(self._add_text(prompt.strip()))
        
        self.messages.append(HumanMessage(content=messages))        

        response = self.chain.invoke(self.messages)

        name = (
            f"{self.get_name()}_output"
            if not hasattr(self, "output")
            else self.output[0] if isinstance(self.output, list) else self.output
        )
        output = Text(name=name, data=response)

        return output


