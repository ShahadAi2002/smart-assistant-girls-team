"""
Command-line interface module.

This module provides the main CLI for the Smart Multi-Modal Assistant.
It allows users to analyze text and images, view recent interactions,
display statistics, and access NLP results through the assistant core.
"""
import os
import sys


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

INTEGRATION_PATH = os.path.join(
    PROJECT_ROOT,
    "integration"
)

KNOWLEDGE_BASE_PATH = os.path.join(
    PROJECT_ROOT,
    "knowledge-base"
)

NLP_MODULE_PATH = os.path.join(
    PROJECT_ROOT,
    "nlp-module"
)


for path in (
    INTEGRATION_PATH,
    KNOWLEDGE_BASE_PATH,
    NLP_MODULE_PATH
):
    if path not in sys.path:
        sys.path.insert(0, path)


from assistant_core import process_input
from knowledge_store import (
    get_recent,
    sentiment_summary,
    top_keywords_overall
)
from summarizer import summarize_text


def analyze_text():
    """
    Analyze text through assistant_core and display:
    intent, sentiment, keywords, and summary.
    """

    text = input(
        "\nEnter the text to analyze: "
    ).strip()

    if not text:
        print("\nError: The text cannot be empty.")
        return

    try:
        print("\nAnalyzing text...")

        results = process_input(
            text,
            input_type="text"
        )

        intent = results["intent"]
        sentiment = results["sentiment"]
        keywords = results["keywords"]

        summary = summarize_text(
            text,
            n_sentences=2
        )

        print("\n==============================")
        print("       Analysis Results")
        print("==============================")

        print("\n1. Intent Classification")
        print("Type:", intent["type"])
        print("Label:", intent["result"]["label"])
        print(
            "Confidence:",
            round(
                float(intent["confidence"]),
                2
            )
        )

        print("\n2. Sentiment Analysis")
        print("Type:", sentiment["type"])
        print("Label:", sentiment["result"]["label"])
        print(
            "Confidence:",
            round(
                float(sentiment["confidence"]),
                2
            )
        )

        print("\n3. Keyword Extraction")
        print("Type:", keywords["type"])
        print(
            "Keywords:",
            keywords["result"]["keywords"]
        )
        print(
            "Confidence:",
            round(
                float(keywords["confidence"]),
                2
            )
        )

        print("\n4. Summary")
        print(summary)

        print("\nText analyzed successfully.")

    except KeyError as error:
        print(
            "\nError: Invalid result format. "
            "Missing key:",
            error
        )

    except (TypeError, ValueError) as error:
        print("\nError:", error)

    except Exception as error:
        print(
            "\nAn error occurred while analyzing the text:",
            error
        )


def analyze_image():
    """
    Analyze an image through assistant_core
    and display face detection results.
    """

    image_path = input(
        "\nEnter the image path: "
    ).strip().strip('"').strip("'")

    if not image_path:
        print(
            "\nError: The image path cannot be empty."
        )
        return

    try:
        print("\nAnalyzing image...")

        result = process_input(
            image_path,
            input_type="image"
        )

        print("\n==============================")
        print("     Face Detection Result")
        print("==============================")

        print("Type:", result["type"])
        print(
            "Face Detected:",
            result["result"]["detected"]
        )
        print(
            "Number of Faces:",
            result["result"]["count"]
        )
        print(
            "Confidence:",
            round(
                float(result["confidence"]),
                2
            )
        )

        if result["result"]["detected"]:
            print("\nFaces detected successfully.")
        else:
            print("\nNo faces were detected.")

    except KeyError as error:
        print(
            "\nError: Invalid face detection result. "
            "Missing key:",
            error
        )

    except (TypeError, ValueError) as error:
        print("\nError:", error)

    except Exception as error:
        print(
            "\nAn error occurred while analyzing the image:",
            error
        )


def show_last_interactions():
    """
    Display recent saved interactions.
    """

    try:
        number_input = input(
            "\nHow many interactions do you want to display? "
        ).strip()

        number = (
            5
            if not number_input
            else int(number_input)
        )

        if number <= 0:
            print(
                "\nError: Please enter a positive number."
            )
            return

        interactions = get_recent(number)

        if not interactions:
            print(
                "\nNo previous interactions were found."
            )
            return

        print("\n==============================")
        print("      Recent Interactions")
        print("==============================")

        for index, interaction in enumerate(
            interactions,
            start=1
        ):
            print(f"\nInteraction {index}:")
            print(interaction)
            print("-" * 40)

    except ValueError:
        print(
            "\nError: Please enter a valid number."
        )

    except Exception as error:
        print(
            "\nAn error occurred while loading interactions:",
            error
        )


def show_statistics():
    """
    Display sentiment statistics and
    the most frequent keywords.
    """

    try:
        sentiment_stats = sentiment_summary()
        top_keywords = top_keywords_overall(
            n=10
        )

        print("\n==============================")
        print("          Statistics")
        print("==============================")

        print("\n1. Sentiment Summary")
        print(
            "Positive:",
            sentiment_stats.get(
                "positive",
                0
            )
        )
        print(
            "Negative:",
            sentiment_stats.get(
                "negative",
                0
            )
        )
        print(
            "Neutral:",
            sentiment_stats.get(
                "neutral",
                0
            )
        )

        print("\n2. Top Keywords")

        if not top_keywords:
            print(
                "No keywords were found."
            )
        else:
            for index, keyword in enumerate(
                top_keywords,
                start=1
            ):
                print(
                    f"{index}. {keyword}"
                )

    except Exception as error:
        print(
            "\nAn error occurred while loading statistics:",
            error
        )


def main_menu():
    """
    Display and manage the CLI menu.
    """

    while True:
        print("\n==============================")
        print("      Smart Assistant Menu")
        print("==============================")
        print("1. Analyze text")
        print("2. Analyze image")
        print("3. Show last interactions")
        print("4. Show statistics")
        print("5. Exit")

        choice = input(
            "\nEnter a number from 1 to 5: "
        ).strip()

        if choice == "1":
            analyze_text()

        elif choice == "2":
            analyze_image()

        elif choice == "3":
            show_last_interactions()

        elif choice == "4":
            show_statistics()

        elif choice == "5":
            print(
                "\nProgram closed successfully."
            )
            break

        elif not choice:
            print(
                "\nError: No option was entered."
            )

        else:
            print(
                "\nInvalid option. "
                "Please enter a number from 1 to 5."
            )


if __name__ == "__main__":
    main_menu()