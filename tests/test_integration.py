import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nlp-module'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'knowledge-base'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'integration'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cv-module'))

from assistant_core import process_input
from knowledge_store import get_recent, sentiment_summary, top_keywords_overall


# text pipeline tests

def test_text_pipeline_output_contract():
    """process_input must return intent, sentiment and keywords for text input."""
    result = process_input("Can you explain how this works?", "text")
    assert "intent" in result
    assert "sentiment" in result
    assert "keywords" in result


def test_text_pipeline_module_output_format():
    """Each module result must follow the unified interface format."""
    result = process_input("Please update my profile settings", "text")
    for key in ("intent", "sentiment", "keywords"):
        assert "type" in result[key]
        assert "result" in result[key]
        assert "confidence" in result[key]
        assert isinstance(result[key]["confidence"], float)


def test_text_interaction_persisted_to_database():
    """Every text interaction must be saved to the interactions table."""
    before = len(get_recent(100))
    process_input("The screen keeps flickering every few minutes", "text")
    after = len(get_recent(100))
    assert after == before + 1


def test_text_interaction_saved_with_correct_type():
    """Saved text interaction must have type field set to text."""
    process_input("Where can I find the export button?", "text")
    recent = get_recent(1)
    assert recent[0][1] == "text"


def test_text_interaction_input_preserved_in_database():
    """The original input text must be stored exactly as provided."""
    text = "Delete all files from the temporary folder"
    process_input(text, "text")
    recent = get_recent(100)
    inputs = [row[2] for row in recent]
    assert text in inputs


# Image Pipeline tests

def test_image_pipeline_output_contract():
    """process_input must return face detection structure for image input."""
    result = process_input("test_image1.jpg", "image")
    assert result["type"] == "face"
    assert "detected" in result["result"]
    assert "count" in result["result"]
    assert isinstance(result["confidence"], float)


def test_image_interaction_persisted_to_database():
    """Every image interaction must be saved to the interactions table."""
    before = len(get_recent(100))
    process_input("test_image1.jpg", "image")
    after = len(get_recent(100))
    assert after == before + 1


def test_image_interaction_saved_with_correct_type():
    """Saved image interaction must have type field set to image."""
    process_input("test_image1.jpg", "image")
    recent = get_recent(100)
    types = [row[1] for row in recent]
    assert "image" in types


#Error Handling tests

def test_nonexistent_image_raises_error_without_saving():
    """Invalid image path must raise ValueError and must not save to database."""
    before = len(get_recent(100))
    with pytest.raises(ValueError):
        process_input("random_photo_xyz.jpg", "image")
    after = len(get_recent(100))
    assert after == before


def test_empty_text_raises_error_without_saving():
    """Empty text must raise ValueError and must not save to database."""
    before = len(get_recent(100))
    with pytest.raises(ValueError):
        process_input("     ", "text")
    after = len(get_recent(100))
    assert after == before


# Statistics tests

def test_sentiment_summary_reflects_saved_data():
    """Sentiment summary must increase after a new text interaction."""
    before = sum(sentiment_summary().values())
    process_input("I absolutely love this new feature", "text")
    after = sum(sentiment_summary().values())
    assert after == before + 1


def test_top_keywords_updates_after_interaction():
    """Top keywords must include words from recent interactions."""
    process_input("The keyboard is not responding properly", "text")
    keywords = top_keywords_overall(20)
    assert isinstance(keywords, list)
    assert len(keywords) > 0

