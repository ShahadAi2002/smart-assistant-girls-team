
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nlp-module'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'knowledge-base'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'integration'))

from assistant_core import process_input
from knowledge_store import get_recent

def test_question_intent():
    result = process_input("How do I open the file?")
    assert result["intent"]["result"]["label"] == "question"
    assert result["intent"]["confidence"] > 0.0

def test_greeting_intent():
    result = process_input("Hello")
    assert result["intent"]["result"]["label"] == "greeting"
    assert result["intent"]["confidence"] > 0.0

def test_command_intent():
    result = process_input("Open the file now")
    assert result["intent"]["result"]["label"] == "command"
    assert result["intent"]["confidence"] > 0.0

def test_complaint_intent():
    result = process_input("The program is not working")
    assert result["intent"]["result"]["label"] == "complaint"
    assert result["intent"]["confidence"] > 0.0


def test_sentiment_positive():
    result = process_input("The service is amazing")
    assert result["sentiment"]["result"]["label"] in ["positive", "negative", "neutral"]
   

def test_keywords_extracted():
    result = process_input("Machine learning and artificial intelligence")
    assert len(result["keywords"]["result"]["keywords"]) > 0
    assert all(isinstance(k, str) for k in result["keywords"]["result"]["keywords"])

def test_sentiment_summary():
    from knowledge_store import sentiment_summary
    summary = sentiment_summary()
    assert all(key in summary for key in ["positive", "negative", "neutral"])
    assert sum(summary.values()) > 0

def test_top_keywords_overall():
    from knowledge_store import top_keywords_overall
    keywords = top_keywords_overall(5)
    assert len(keywords) <= 5
    assert all(isinstance(k, str) for k in keywords)

