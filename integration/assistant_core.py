import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "nlp-module"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "knowledge-base"))

from intent_classifier import classify_intent
from knowledge_store import log_intent, save_interaction


def process_input(text):
    intent_result = classify_intent(text)
    
    log_intent(
        text=text,
        intent=intent_result["result"]["label"],
        confidence=intent_result["confidence"]
    )
    
    save_interaction(
        type="text",
        input_data=text,
        output_json=str(intent_result)
    )
    
    return intent_result

# def classify_intent(text):
#     return {
#         "type": "intent",
#         "result": {"intent": "unknown"},
#         "confidence": 0.0
#     }

def analyze_sentiment(text):
    return {
        "type": "sentiment",
        "result": {"label": "neutral"},
        "confidence": 0.0
    }

def extract_keywords(text):
    return {
        "type": "keywords",
        "result": {"keywords": []},
        "confidence": 0.0
    }

def detect_faces(image_path):
    return {
        "type": "face",
        "result": {"detected": False, "count": 0},
        "confidence": 0.0
    }