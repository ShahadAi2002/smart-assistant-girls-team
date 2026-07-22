import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "nlp-module"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "knowledge-base"))

from intent_classifier import classify_intent
from sentiment_analyzer import analyze_sentiment
from keyword_extractor import extract_keywords
from knowledge_store import log_intent,log_sentiment, save_interaction


def process_input(text):
    
    intent_result = classify_intent(text)
    sentiment_result = analyze_sentiment(text)
    keyword_result = extract_keywords(text)
    
    log_intent(
        text=text,
        intent=intent_result["result"]["label"],
        confidence=intent_result["confidence"]
    )

    log_sentiment(
        text=text,
        label=sentiment_result["result"]["label"],
        confidence=sentiment_result["confidence"]
    )
    
    save_interaction(
        type="text",
        input_data=text,
        output_json=str({
            "intent": intent_result,
            "sentiment": sentiment_result,
            "keywords": keyword_result
        })
    )
    
    return {
        "intent": intent_result,
        "sentiment": sentiment_result,
        "keywords": keyword_result
    }



