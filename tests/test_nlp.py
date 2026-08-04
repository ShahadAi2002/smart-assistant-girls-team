import os
import sys


# Add the nlp-module folder to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

NLP_MODULE_PATH = os.path.join(
    PROJECT_ROOT,
    "nlp-module"
)

if NLP_MODULE_PATH not in sys.path:
    sys.path.insert(0, NLP_MODULE_PATH)


from intent_classifier import classify_intent
from sentiment_analyzer import analyze_sentiment
from keyword_extractor import extract_keywords
from summarizer import summarize


# =========================================================
# Intent Tests
# =========================================================

def test_intent_greeting():
    result = classify_intent(
        "Good morning, nice to meet you."
    )

    assert result["type"] == "intent"
    assert result["result"]["label"] == "greeting"
    assert 0.0 <= result["confidence"] <= 1.0


def test_intent_complaint():
    result = classify_intent(
        "The application keeps crashing."
    )

    assert result["type"] == "intent"
    assert result["result"]["label"] == "complaint"
    assert 0.0 <= result["confidence"] <= 1.0


# =========================================================
# Sentiment Tests
# =========================================================

def test_sentiment_positive():
    result = analyze_sentiment(
        "The service was excellent and helpful."
    )

    assert result["type"] == "sentiment"
    assert result["result"]["label"] == "positive"
    assert 0.0 <= result["confidence"] <= 1.0


def test_sentiment_negative():
    result = analyze_sentiment(
        "The application is terrible and frustrating."
    )

    assert result["type"] == "sentiment"
    assert result["result"]["label"] == "negative"
    assert 0.0 <= result["confidence"] <= 1.0


# =========================================================
# Keyword Extraction Tests
# =========================================================

def test_keyword_extractor_returns_dictionary():
    result = extract_keywords(
        "Artificial intelligence improves modern technology."
    )

    assert isinstance(result, dict)
    assert result["type"] == "keywords"
    assert "keywords" in result["result"]


def test_keyword_extractor_returns_keywords():
    result = extract_keywords(
        "Python is used for machine learning and data analysis."
    )

    keywords = result["result"]["keywords"]

    assert isinstance(keywords, list)
    assert len(keywords) > 0
    assert 0.0 <= result["confidence"] <= 1.0


# =========================================================
# Summarizer Tests
# =========================================================

def test_summarizer_returns_string():
    text = (
        "Artificial intelligence is growing rapidly. "
        "It is used in healthcare and education. "
        "Machine learning is an important part of AI. "
        "Many companies use AI to improve their services."
    )

    result = summarize(
        text,
        n_sentences=2
    )

    assert isinstance(result, str)
    assert len(result.strip()) > 0


def test_summarizer_returns_requested_sentences():
    text = (
        "Python is a popular programming language. "
        "It is commonly used in data science. "
        "Python has many useful libraries. "
        "It is also used in artificial intelligence."
    )

    result = summarize(
        text,
        n_sentences=2
    )

    sentence_count = result.count(".")
    assert sentence_count <= 2