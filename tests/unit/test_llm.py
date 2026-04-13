""" Tests for collectra.llms module """

import pytest
from unittest.mock import MagicMock, patch
from collectra.tasks.llms import LLM
from collectra.types import Text, Image

def test_add_text():
    mock_self = MagicMock(spec=LLM)
    result = LLM._add_text(mock_self,"test text")
    assert result == {"type": "text", "text": "test text"}

def test_add_content():
    mock_self = MagicMock(spec=LLM)

    # Test Text branch
    mock_self._add_text.return_value = {"type": "text", "text": "hello"}
    mock_text = MagicMock(spec=Text)
    mock_text.return_value = "hello"

    result = LLM._add_content(mock_self, mock_text)
    assert result == {"type": "text", "text": "hello"}

    # Test Image branch
    mock_self.image_content.return_value = {"type": "image", "data": "iiVBORw0KGgoAAAANSUhEUgAAAAUA"}
    mock_image = MagicMock(spec=Image)
    result = LLM._add_content(mock_self, mock_image)
    assert result == {"type": "image", "data": "iiVBORw0KGgoAAAANSUhEUgAAAAUA"}

def test_image_content():
    mock_self = MagicMock(spec=LLM)
    mock_self.llm = MagicMock()

    mock_image = MagicMock(spec=Image)
    mock_image.get_encoding.return_value = "iiVBORw0KGgoAAAANSUhEUgAAAAUA"
    mock_image.mime.return_value = "image/png"

    with patch('collectra.tasks.llms.llmloader.LLMWrapper.format') as mock_format:
        mock_format.return_value = {
            "type": "image",
            "data": "iiVBORw0KGgoAAAANSUhEUgAAAAUA",
            "mime_type": "image/png",
        }

        result = LLM.image_content(mock_self, mock_image)
        
        mock_format.assert_called_once_with(
            mock_self.llm,
            data_type="image",
            data={
                "data": "iiVBORw0KGgoAAAANSUhEUgAAAAUA",
                "mime_type": "image/png",
            },
        )       
    
    assert result == {
            "type": "image",
            "data": "iiVBORw0KGgoAAAANSUhEUgAAAAUA",
            "mime_type": "image/png",
        }

