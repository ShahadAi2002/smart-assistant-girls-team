import os
import sys

# Add NLP module path
NLP_MODULE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "nlp-module"
    )
)

# Add knowledge base path
KNOWLEDGE_BASE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "knowledge-base"
    )
)

if NLP_MODULE_PATH not in sys.path:
    sys.path.append(NLP_MODULE_PATH)

if KNOWLEDGE_BASE_PATH not in sys.path:
    sys.path.append(KNOWLEDGE_BASE_PATH)


from intent_classifier import classify_intent
from sentiment_analyzer import analyze_sentiment
from keyword_extractor import extract_keywords
from summarizer import summarize

from knowledge_store import (
    log_intent,
    log_sentiment,
    save_interaction
)


def process_input(text):
    """
    Analyze the input text and return all NLP results.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Input text cannot be empty.")

    # Run NLP functions
    intent_result = classify_intent(text)
    sentiment_result = analyze_sentiment(text)
    keyword_result = extract_keywords(text)
    summary_result = summarize(text)

    # Combine all results
    results = {
        "intent": intent_result,
        "sentiment": sentiment_result,
        "keywords": keyword_result,
        "summary": summary_result
    }

    # Store intent result
    log_intent(
        text=text,
        intent=intent_result["result"]["label"],
        confidence=float(intent_result["confidence"])
    )

    # Store sentiment result
    log_sentiment(
        text=text,
        label=sentiment_result["result"]["label"],
        confidence=float(sentiment_result["confidence"])
    )

    # Save complete interaction
    save_interaction(
        type="text",
        input_data=text,
        output_json=str(results)
    )

    return results