import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "nlp-module"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "knowledge-base"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cv-module"))

from intent_classifier import classify_intent
from sentiment_analyzer import analyze_sentiment
from keyword_extractor import extract_keywords
from face_detector import detect_faces
from knowledge_store import log_intent, log_sentiment, save_interaction


def process_input(input_data, input_type="text"):
    """
    Process text or image input through the assistant pipeline.

    Args:
        input_data (str): The input text or image path.
        input_type (str): Type of input - "text" or "image". Defaults to "text".

    Returns:
        dict: Results from the NLP pipeline (text) or face detection (image).

    Raises:
        ValueError: If input_data is empty or input_type is invalid.
        TypeError: If input_data is not a string.
    """

    if not isinstance(input_data, str):
        raise TypeError("input_data must be a string.")

    input_data = input_data.strip()

    if not input_data:
        raise ValueError("input_data cannot be empty.")

    if input_type not in ("text", "image"):
        raise ValueError("input_type must be 'text' or 'image'.")

    if input_type == "text":
        intent_result = classify_intent(input_data)
        sentiment_result = analyze_sentiment(input_data)
        keyword_result = extract_keywords(input_data)

        log_intent(
            text=input_data,
            intent=intent_result["result"]["label"],
            confidence=intent_result["confidence"]
        )

        log_sentiment(
            text=input_data,
            label=sentiment_result["result"]["label"],
            confidence=sentiment_result["confidence"]
        )

        save_interaction(
            type="text",
            input_data=input_data,
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

    elif input_type == "image":
        if not os.path.exists(input_data):
            raise ValueError(f"Image file not found: {input_data}")

        face_result = detect_faces(input_data)

        save_interaction(
            type="image",
            input_data=input_data,
            output_json=str(face_result)
        )

        return face_result 



