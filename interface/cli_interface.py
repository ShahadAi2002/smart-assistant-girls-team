import os
import sys

# Add integration folder path
INTEGRATION_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "integration"
    )
)

# Add knowledge-base folder path
KNOWLEDGE_BASE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "knowledge-base"
    )
)

if INTEGRATION_PATH not in sys.path:
    sys.path.append(INTEGRATION_PATH)

if KNOWLEDGE_BASE_PATH not in sys.path:
    sys.path.append(KNOWLEDGE_BASE_PATH)


from assistant_core import process_input
from knowledge_store import get_recent


def analyze_text():
    """
    Analyze text using assistant_core.process_input().
    """

    text = input("\nEnter the text to analyze: ").strip()

    if not text:
        print("\nError: The text cannot be empty.")
        return

    try:
        print("\nAnalyzing text...")

        result = process_input(text)

        intent = result["intent"]
        sentiment = result["sentiment"]
        keywords = result["keywords"]
        summary = result["summary"]

        print("\n==============================")
        print("       Analysis Results")
        print("==============================")

        print("\n1. Intent Classification")
        print("Type:", intent["type"])
        print("Label:", intent["result"]["label"])
        print(
            "Confidence:",
            round(float(intent["confidence"]), 2)
        )

        print("\n2. Sentiment Analysis")
        print("Type:", sentiment["type"])
        print("Label:", sentiment["result"]["label"])
        print(
            "Confidence:",
            round(float(sentiment["confidence"]), 2)
        )

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
            "\nError: One of the functions returned "
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

    print("\nImage analysis is not available yet.")


def show_last_interactions():
    """
    Display previous interactions from knowledge_store.
    """

    try:
        n = int(
            input("\nHow many interactions do you want to show? ")
        )

        if n <= 0:
            print("\nPlease enter a number greater than zero.")
            return

        interactions = get_recent(n)

        if not interactions:
            print("\nNo previous interactions found.")
            return

        print("\n==============================")
        print("     Last Interactions")
        print("==============================")

        for interaction in interactions:
            print("\n", interaction)

    except ValueError:
        print("\nPlease enter a valid number.")

    except Exception as error:
        print(
            "\nAn error occurred while retrieving interactions:",
            error
        )


def show_statistics():
    """
    Temporary statistics function.
    """

    print("\nStatistics are not available yet.")


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
            print("\nError: No option was entered.")

        else:
            print(
                "\nInvalid option. "
                "Please enter a number from 1 to 5."
            )


if __name__ == "__main__":
    main_menu()
    