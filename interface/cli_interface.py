import os
import sys


# Project root path
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


# Module paths
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


# Add module paths to sys.path
for path in (
    INTEGRATION_PATH,
    KNOWLEDGE_BASE_PATH,
    NLP_MODULE_PATH
):
    if path not in sys.path:
        sys.path.append(path)


# Imports
from assistant_core import process_input
from knowledge_store import get_recent
from summarizer import summarize_text
import summarizer

print("Summarizer loaded from:", summarizer.__file__)

def analyze_text():
    """
    Analyze text using assistant_core.process_input()
    and display intent, sentiment, keywords, and summary.
    """

    text = input(
        "\nEnter the text to analyze: "
    ).strip()

    if not text:
        print(
            "\nError: The text cannot be empty."
        )
        return

    try:
        print(
            "\nAnalyzing text..."
        )

        # Get intent, sentiment, and keywords
        # from assistant_core
        results = process_input(
            text,
            input_type="text"
        )

        intent = results["intent"]
        sentiment = results["sentiment"]
        keywords = results["keywords"]

        # Generate an extractive summary
        summary = summarize_text(
            text,
            n_sentences=2
        )

        print(
            "\n=============================="
        )
        print(
            "       Analysis Results"
        )
        print(
            "=============================="
        )

        # Intent result
        print(
            "\n1. Intent Classification"
        )
        print(
            "Type:",
            intent["type"]
        )
        print(
            "Label:",
            intent["result"]["label"]
        )
        print(
            "Confidence:",
            round(
                float(intent["confidence"]),
                2
            )
        )

        # Sentiment result
        print(
            "\n2. Sentiment Analysis"
        )
        print(
            "Type:",
            sentiment["type"]
        )
        print(
            "Label:",
            sentiment["result"]["label"]
        )
        print(
            "Confidence:",
            round(
                float(sentiment["confidence"]),
                2
            )
        )

        # Keyword result
        print(
            "\n3. Keyword Extraction"
        )
        print(
            "Type:",
            keywords["type"]
        )
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

        # Summary result
        print(
            "\n4. Summary"
        )
        print(
            summary
        )

        print(
            "\nText analyzed successfully."
        )

    except KeyError as error:
        print(
            "\nError: Invalid result format. "
            "Missing key:",
            error
        )

    except TypeError as error:
        print(
            "\nError: Invalid data type:",
            error
        )

    except ValueError as error:
        print(
            "\nError:",
            error
        )

    except Exception as error:
        print(
            "\nAn error occurred while "
            "analyzing the text:",
            error
        )


def analyze_image():
    """
    Analyze an image using assistant_core.
    """

    image_path = input(
        "\nEnter the image path: "
    ).strip()

    if not image_path:
        print(
            "\nError: The image path cannot be empty."
        )
        return

    try:
        result = process_input(
            image_path,
            input_type="image"
        )

        print(
            "\nImage Analysis Result:"
        )
        print(
            result
        )

    except Exception as error:
        print(
            "\nAn error occurred while "
            "analyzing the image:",
            error
        )


def show_last_interactions():
    """
    Display recent interactions using
    knowledge_store.get_recent(n).
    """

    try:
        number_input = input(
            "\nHow many interactions do you want to display? "
        ).strip()

        if not number_input:
            number = 5
        else:
            number = int(
                number_input
            )

        if number <= 0:
            print(
                "\nError: Please enter a positive number."
            )
            return

        interactions = get_recent(
            number
        )

        if not interactions:
            print(
                "\nNo previous interactions were found."
            )
            return

        print(
            "\n=============================="
        )
        print(
            "      Recent Interactions"
        )
        print(
            "=============================="
        )

        for index, interaction in enumerate(
            interactions,
            start=1
        ):
            print(
                f"\nInteraction {index}:"
            )
            print(
                interaction
            )
            print(
                "-" * 40
            )

    except ValueError:
        print(
            "\nError: Please enter a valid number."
        )

    except Exception as error:
        print(
            "\nAn error occurred while "
            "loading recent interactions:",
            error
        )


def show_statistics():
    """
    Temporary function for displaying statistics.
    """

    print(
        "\nStatistics are not available yet."
    )


def main_menu():
    """
    Display and manage the main CLI menu.
    """

    while True:
        print(
            "\n=============================="
        )
        print(
            "      Smart Assistant Menu"
        )
        print(
            "=============================="
        )
        print(
            "1. Analyze text"
        )
        print(
            "2. Analyze image"
        )
        print(
            "3. Show last interactions"
        )
        print(
            "4. Show statistics"
        )
        print(
            "5. Exit"
        )

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