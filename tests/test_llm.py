from pathlib import Path
from collectra.pipeline import CollectraManager


TEST_DATA = Path(__file__).parent / "data"

def test_llm():
    workflow = CollectraManager.load(TEST_DATA / "basic_llm_pipeline.yaml")
    
    result = workflow("love")
    assert result == "Write a poem about love in haiku"