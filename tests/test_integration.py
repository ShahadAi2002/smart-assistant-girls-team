
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
    assert result["type"] == "intent"
    assert result["result"]["label"] == "question"
    assert result["confidence"] > 0.0

def test_greeting_intent():
    result = process_input("Hello")
    assert result["type"] == "intent"
    assert result["result"]["label"] == "greeting"
    assert result["confidence"] > 0.0

def test_command_intent():
    result = process_input("Open the file now")
    assert result["type"] == "intent"
    assert result["result"]["label"] == "command"
    assert result["confidence"] > 0.0

def test_complaint_intent():
    result = process_input("The program is not working")
    assert result["type"] == "intent"
    assert result["result"]["label"] == "complaint"
    assert result["confidence"] > 0.0