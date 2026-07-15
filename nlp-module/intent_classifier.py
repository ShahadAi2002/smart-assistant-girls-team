# Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Dataset: 60 sentences divided into 4 classes
texts = [
    # Question
    "What is the weather today?",
    "How can I use the application?",
    "When does the meeting start?",
    "Where can I find the file?",
    "Can you help me?",
    "How many students are there?",
    "What does artificial intelligence mean?",
    "Is there a new update?",
    "Why is the program not working?",
    "How do I save the file?",
    "Where is the settings menu?",
    "Can I change my password?",
    "What time is the class?",
    "How do I upload an image?",
    "Is this feature available?",

    # Command
    "Open the file now",
    "Save the changes",
    "Run the program",
    "Send the report",
    "Write the summary",
    "Show the results",
    "Delete the message",
    "Start the analysis",
    "Organize the data",
    "Close the window",
    "Update the application",
    "Create a new folder",
    "Print the document",
    "Search for the keyword",
    "Restart the system",

    # Complaint
    "The program is not working",
    "The application is very slow",
    "I cannot log in",
    "There is an error in the system",
    "The service is bad",
    "The page does not open",
    "The results are incorrect",
    "The sound is not working",
    "The connection keeps disconnecting",
    "The interface is difficult to use",
    "The file does not upload",
    "The application crashes often",
    "The response takes too long",
    "The button is not working",
    "The system freezes frequently",

    # Greeting
    "Hello",
    "Hi",
    "Good morning",
    "Good evening",
    "How are you?",
    "Nice to meet you",
    "Welcome",
    "Hey",
    "Greetings",
    "Have a nice day",
    "Good afternoon",
    "Hope you are doing well",
    "Nice to see you",
    "Hello everyone",
    "Good to hear from you"
]


labels = [
    # Question
    "question", "question", "question", "question", "question",
    "question", "question", "question", "question", "question",
    "question", "question", "question", "question", "question",

    # Command
    "command", "command", "command", "command", "command",
    "command", "command", "command", "command", "command",
    "command", "command", "command", "command", "command",

    # Complaint
    "complaint", "complaint", "complaint", "complaint", "complaint",
    "complaint", "complaint", "complaint", "complaint", "complaint",
    "complaint", "complaint", "complaint", "complaint", "complaint",

    # Greeting
    "greeting", "greeting", "greeting", "greeting", "greeting",
    "greeting", "greeting", "greeting", "greeting", "greeting",
    "greeting", "greeting", "greeting", "greeting", "greeting"
]


# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(texts)
y = labels


# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# Model 1: Naive Bayes
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)

nb_predictions = naive_bayes_model.predict(X_test)
nb_accuracy = accuracy_score(
    y_test,
    nb_predictions
)


# Model 2: Logistic Regression
logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(
    X_train,
    y_train
)

lr_predictions = logistic_model.predict(X_test)
lr_accuracy = accuracy_score(
    y_test,
    lr_predictions
)


# Choose the best model based on accuracy
if lr_accuracy >= nb_accuracy:
    best_model = logistic_model
    best_model_name = "Logistic Regression"
else:
    best_model = naive_bayes_model
    best_model_name = "Naive Bayes"


def classify_intent(text):
    """
    Classify the text intent and return
    the result using the unified output format.
    """

    # Validate the input
    if not isinstance(text, str):
        raise TypeError("The input text must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("The input text cannot be empty.")

    # Convert the input text into TF-IDF features
    text_vector = vectorizer.transform([text])

    # Predict the intent
    predicted_intent = best_model.predict(
        text_vector
    )[0]

    # Get prediction probabilities
    probabilities = best_model.predict_proba(
        text_vector
    )[0]

    confidence = probabilities.max()

    return {
        "type": "intent",
        "result": {
            "label": str(predicted_intent)
        },
        "confidence": round(
            float(confidence),
            2
        )
    }


# Run tests only when this file is executed directly
if __name__ == "__main__":

    print(
        "Naive Bayes Accuracy:",
        nb_accuracy
    )

    print(
        "Logistic Regression Accuracy:",
        lr_accuracy
    )

    print(
        "\nClassification Report "
        "for Naive Bayes:"
    )

    print(
        classification_report(
            y_test,
            nb_predictions,
            zero_division=0
        )
    )

    print(
        "\nClassification Report "
        "for Logistic Regression:"
    )

    print(
        classification_report(
            y_test,
            lr_predictions,
            zero_division=0
        )
    )

    print(
        "\nDataset size:",
        len(texts)
    )

    print(
        "Number of labels:",
        len(labels)
    )

    print(
        "TF-IDF shape:",
        X.shape
    )

    print(
        "Classes:",
        set(y)
    )

    sample_text = "How do I open the file?"
    result = classify_intent(sample_text)

    print(
        "\nBest Model:",
        best_model_name
    )

    print(
        "Sample text:",
        sample_text
    )

    print(
        "Result:",
        result
    )