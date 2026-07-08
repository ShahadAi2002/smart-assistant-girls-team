from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# Point 1: Create a manual dataset

texts = [
    # Positive sentences
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

    # Negative sentences
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

    # Neutral sentences
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
    "The program runs on the computer"
]

labels = (
    ["positive"] * 15
    + ["negative"] * 15
    + ["neutral"] * 15
)


# Point 2: Create a Logistic Regression model

sentiment_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2)
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


# Point 3: Evaluate using 5-fold cross-validation

scores = cross_val_score(
    sentiment_model,
    texts,
    labels,
    cv=5,
    scoring="accuracy"
)

print("Cross-validation scores:", scores)
print("Average accuracy:", scores.mean())


# Train the final model using all data

sentiment_model.fit(texts, labels)


def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {
            "type": "sentiment",
            "result": {
                "label": "neutral"
            },
            "confidence": 0.0
        }

    prediction = str(sentiment_model.predict([text])[0])

    probabilities = sentiment_model.predict_proba([text])[0]

    confidence = float(probabilities.max())

    return {
        "type": "sentiment",
        "result": {
            "label": prediction
        },
        "confidence": confidence
    }

# Test the function

if __name__ == "__main__":
    test_sentences = [
        "The service is amazing",
        "The application is very bad",
        "The meeting starts at nine"
    ]

    for sentence in test_sentences:
        result = analyze_sentiment(sentence)

        print("Text:", sentence)
        print("Result:", result)
        print("-" * 30)