from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# Dataset: 60 sentences divided equally into 3 classes
texts = [
    # -------------------------
    # Positive sentences: 20
    # -------------------------
    "The service is excellent",
    "I am very happy today",
    "The application is easy to use",
    "The experience was wonderful",
    "I really like this product",
    "The employees are helpful",
    "The result is amazing",
    "The project was successful",
    "I feel comfortable and happy",
    "The performance is excellent",
    "The idea is creative",
    "The service was fast",
    "This is a beautiful day",
    "I enjoyed the lesson",
    "The work was completed perfectly",
    "I am satisfied with the service",
    "This product is very good",
    "The result is acceptable and useful",
    "I enjoy using this application",
    "The team did a great job",

    # -------------------------
    # Negative sentences: 20
    # -------------------------
    "The service is terrible",
    "I am very sad today",
    "The application is difficult to use",
    "The experience was awful",
    "I do not like this product",
    "The employees are unhelpful",
    "The result is disappointing",
    "The project failed",
    "I feel tired and upset",
    "The performance is poor",
    "The idea is useless",
    "The service was very slow",
    "This is a bad day",
    "I hated the lesson",
    "The work contains many mistakes",
    "I am not happy",
    "I am not satisfied with the service",
    "This product is not good",
    "The result is not acceptable",
    "I do not enjoy using this application",

    # -------------------------
    # Neutral sentences: 20
    # -------------------------
    "The meeting starts at ten",
    "The file was sent today",
    "The product is available",
    "The lesson starts in the morning",
    "The system was updated",
    "The office is on the second floor",
    "The exam is on Sunday",
    "The request is under review",
    "The lecture lasts one hour",
    "The user logged into the system",
    "The car is outside the house",
    "The report contains five pages",
    "The class has twenty students",
    "The message was delivered",
    "The program runs on the computer",
    "The meeting room is number five",
    "The application was installed yesterday",
    "The document is stored in the folder",
    "The course contains six lessons",
    "The computer is connected to the network"
]


# Labels must match the order and number of sentences
labels = (
    ["positive"] * 20
    + ["negative"] * 20
    + ["neutral"] * 20
)


# Create a pipeline containing TF-IDF and Logistic Regression
sentiment_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


# Evaluate the model using 5-fold cross-validation
scores = cross_val_score(
    sentiment_model,
    texts,
    labels,
    cv=5,
    scoring="accuracy"
)


# Train the final model using all dataset sentences
sentiment_model.fit(texts, labels)


def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text.

    The function returns one of three labels:
    positive, negative, or neutral.

    Output format:
    {
        "type": "sentiment",
        "result": {
            "label": "positive"
        },
        "confidence": 0.85
    }
    """

    # Check that the input is a string
    if not isinstance(text, str):
        raise TypeError(
            "The input text must be a string."
        )

    # Remove unnecessary spaces
    text = text.strip()

    # Reject empty input
    if not text:
        raise ValueError(
            "The input text cannot be empty."
        )

    # Predict the sentiment label
    prediction = sentiment_model.predict(
        [text]
    )[0]

    # Get the prediction probabilities
    probabilities = sentiment_model.predict_proba(
        [text]
    )[0]

    # Select the highest probability as confidence
    confidence = probabilities.max()

    # Return the unified output format
    return {
        "type": "sentiment",
        "result": {
            "label": str(prediction)
        },
        "confidence": round(
            float(confidence),
            2
        )
    }


# Run evaluation and tests only when this file is executed directly
if __name__ == "__main__":

    print(
        "Dataset size:",
        len(texts)
    )

    print(
        "Number of labels:",
        len(labels)
    )

    print(
        "Cross-validation scores:",
        scores
    )

    print(
        "Average accuracy:",
        round(float(scores.mean()), 2)
    )

    test_sentences = [
        "The service is amazing",
        "The application is very bad",
        "The meeting starts at nine",
        "I am not happy",
        "I am satisfied with the result",
        "This product is not good"
    ]

    print("\nSentiment Test Results")

    for sentence in test_sentences:
        result = analyze_sentiment(sentence)

        print("\nText:", sentence)
        print("Result:", result)
        print("-" * 40)