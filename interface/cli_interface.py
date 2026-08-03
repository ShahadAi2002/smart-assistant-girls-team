import os
import sys

# Add the nlp-module folder path so Python can import NLP files
NLP_MODULE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "nlp-module"
    )
)

if NLP_MODULE_PATH not in sys.path:
    sys.path.append(NLP_MODULE_PATH)


from intent_classifier import classify_intent
from sentiment_analyzer import analyze_sentiment
from keyword_extractor import extract_keywords
from summarizer import summarize_text


def analyze_text():
    """
    Analyze text using intent classification,
    sentiment analysis, keyword extraction,
    and the temporary summarizer function.
    """

    text = input("\nEnter the text to analyze: ").strip()

    # Handle empty text
    if not text:
        print("\nError: The text cannot be empty.")
        return

    try:
        print("\nAnalyzing text...")

        # Run the real NLP functions
        intent = classify_intent(text)
        sentiment = analyze_sentiment(text)
        keywords = extract_keywords(text)

        # Temporary summarizer function
        summary = summarize_text(text)

        print("\n==============================")
        print("       Analysis Results")
        print("==============================")

        # Intent result
        print("\n1. Intent Classification")
        print("Type:", intent["type"])
        print("Label:", intent["result"]["label"])
        print(
            "Confidence:",
            round(float(intent["confidence"]), 2)
        )

        # Sentiment result
        print("\n2. Sentiment Analysis")
        print("Type:", sentiment["type"])
        print("Label:", sentiment["result"]["label"])
        print(
            "Confidence:",
            round(float(sentiment["confidence"]), 2)
        )

        # Keyword result
        print("\n3. Keyword Extraction")
        print("Type:", keywords["type"])
        print(
            "Keywords:",
            keywords["result"]["keywords"]
        )
        print(
            "Confidence:",
            round(float(keywords["confidence"]), 2)
        )

        # Summary result
        print("\n4. Summary")
        print(summary)

        print("\nText analyzed successfully.")

    except KeyError as error:
        print(
            "\nError: Invalid result format. Missing key:",
            error
        )

    except TypeError as error:
        print(
            "\nError: One of the NLP functions returned "
            "an invalid data type:",
            error
        )

    except Exception as error:
        print(
            "\nAn error occurred while analyzing the text:",
            error
        )


def analyze_image():
    """
    Temporary image analysis function.
    """

    print(
        "\nImage analysis is not available yet."
    )


def show_last_interactions():
    """
    Temporary function for displaying previous interactions.
    """

    print(
        "\nPrevious interactions are not available yet."
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
            print("\nProgram closed successfully.")
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