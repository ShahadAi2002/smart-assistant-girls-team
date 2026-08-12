"""
Sentiment analysis module.

This module builds and evaluates a sentiment classification model
for classifying text into positive, negative, or neutral sentiment.

It provides the analyze_sentiment() function, which returns
the predicted sentiment label and confidence score.
"""
# Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report
)


def combine_parts(first_parts, second_parts):
    """
    Combine two lists to create unique sentences.

    10 first parts × 10 second parts = 100 sentences.
    """

    return [
        f"{first_part} {second_part}".strip()
        for first_part in first_parts
        for second_part in second_parts
    ]


# =========================================================
# Positive Dataset: 100 sentences
# =========================================================

positive_starters = [
    "I am happy because",
    "I am pleased that",
    "It is wonderful that",
    "I really appreciate how",
    "I am impressed because",
    "It is great that",
    "I feel satisfied because",
    "I am excited that",
    "I am glad because",
    "I enjoyed how"
]


positive_endings = [
    "the application works smoothly.",
    "the service responded quickly.",
    "the team completed the task successfully.",
    "the results were accurate and useful.",
    "the interface was easy to use.",
    "the problem was solved immediately.",
    "the project achieved all its goals.",
    "the employees were friendly and helpful.",
    "the new feature saved a lot of time.",
    "the experience was better than expected."
]


positive_sentences = combine_parts(
    positive_starters,
    positive_endings
)


# =========================================================
# Negative Dataset: 100 sentences
# =========================================================

negative_starters = [
    "I am disappointed because",
    "I am unhappy that",
    "It is frustrating that",
    "I really dislike how",
    "I am upset because",
    "It is terrible that",
    "I feel dissatisfied because",
    "I regret that",
    "I am annoyed because",
    "I hated how"
]


negative_endings = [
    "the application keeps crashing.",
    "the service responds very slowly.",
    "the team failed to complete the task.",
    "the results were inaccurate and useless.",
    "the interface was difficult to use.",
    "the problem was not solved.",
    "the project failed to achieve its goals.",
    "the employees were rude and unhelpful.",
    "the new feature wasted a lot of time.",
    "the experience was worse than expected."
]


negative_sentences = combine_parts(
    negative_starters,
    negative_endings
)


# =========================================================
# Neutral Dataset: 100 sentences
# =========================================================

neutral_starters = [
    "The system reports that",
    "The application shows that",
    "The document states that",
    "The user confirmed that",
    "The report indicates that",
    "The database records that",
    "The program displays that",
    "The message says that",
    "The schedule shows that",
    "The file confirms that"
]


neutral_endings = [
    "the meeting starts at ten.",
    "the file was uploaded yesterday.",
    "the system was updated this morning.",
    "the report contains five pages.",
    "the course includes six lessons.",
    "the application is connected to the network.",
    "the request is currently under review.",
    "the document is stored in the main folder.",
    "the test contains twenty questions.",
    "the session ended at three o'clock."
]


neutral_sentences = combine_parts(
    neutral_starters,
    neutral_endings
)


# =========================================================
# Create Complete Dataset
# =========================================================

sentiment_data = {
    "positive": positive_sentences,
    "negative": negative_sentences,
    "neutral": neutral_sentences
}


texts = []
labels = []


for sentiment_label, sentiment_sentences in sentiment_data.items():
    texts.extend(
        sentiment_sentences
    )

    labels.extend(
        [sentiment_label] * len(sentiment_sentences)
    )


# =========================================================
# Verify Dataset Size and Balance
# =========================================================

assert len(positive_sentences) == 100
assert len(negative_sentences) == 100
assert len(neutral_sentences) == 100

assert len(texts) == 300
assert len(labels) == 300

assert labels.count("positive") == 100
assert labels.count("negative") == 100
assert labels.count("neutral") == 100


# =========================================================
# Split Dataset into Training and Testing Data
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)


# =========================================================
# Baseline Pipeline
# Before GridSearchCV
# =========================================================

baseline_sentiment_model = Pipeline([
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
            max_iter=3000,
            random_state=42
        )
    )
])


# =========================================================
# Baseline 5-Fold Cross-Validation
# =========================================================

baseline_cv_scores = cross_val_score(
    baseline_sentiment_model,
    texts,
    labels,
    cv=5,
    scoring="accuracy"
)


# =========================================================
# Train and Evaluate Baseline Model
# =========================================================

baseline_sentiment_model.fit(
    X_train,
    y_train
)


baseline_predictions = baseline_sentiment_model.predict(
    X_test
)


baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)


# =========================================================
# GridSearchCV Pipeline
# =========================================================

grid_pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=3000,
            random_state=42
        )
    )
])


# =========================================================
# GridSearchCV Parameters
# =========================================================

param_grid = {
    "tfidf__ngram_range": [
        (1, 1),
        (1, 2)
    ],

    "tfidf__min_df": [
        1,
        2
    ],

    "classifier__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],

    # lbfgs supports multiclass classification
    # and avoids the previous convergence warnings
    "classifier__solver": [
        "lbfgs"
    ],

    "classifier__class_weight": [
        None,
        "balanced"
    ]
}


# =========================================================
# Run GridSearchCV
# =========================================================

grid_search = GridSearchCV(
    estimator=grid_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    error_score="raise"
)


grid_search.fit(
    X_train,
    y_train
)


# =========================================================
# Optimized Model
# After GridSearchCV
# =========================================================

optimized_sentiment_model = grid_search.best_estimator_


optimized_predictions = optimized_sentiment_model.predict(
    X_test
)


optimized_accuracy = accuracy_score(
    y_test,
    optimized_predictions
)


# =========================================================
# Compare Baseline and Optimized Models
# =========================================================

if optimized_accuracy > baseline_accuracy:
    sentiment_model = optimized_sentiment_model
    best_model_name = "Optimized Logistic Regression"
    best_model_accuracy = optimized_accuracy
else:
    sentiment_model = baseline_sentiment_model
    best_model_name = "Baseline Logistic Regression"
    best_model_accuracy = baseline_accuracy


# Train the selected model using the complete dataset
sentiment_model.fit(
    texts,
    labels
)


def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text.

    Args:
        text (str): Input text.

    Returns:
        dict: Sentiment label and confidence.

    Example:
        {
            "type": "sentiment",
            "result": {
                "label": "positive"
            },
            "confidence": 0.85
        }
    """

    if not isinstance(text, str):
        raise TypeError(
            "The input text must be a string."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "The input text cannot be empty."
        )

    prediction = sentiment_model.predict(
        [text]
    )[0]

    probabilities = sentiment_model.predict_proba(
        [text]
    )[0]

    confidence = probabilities.max()

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


# =========================================================
# Run Evaluation
# =========================================================

if __name__ == "__main__":

    print(
        "============================================"
    )

    print(
        "       Sentiment Model Evaluation"
    )

    print(
        "============================================"
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
        "\nSentences in each class:"
    )

    print(
        "Positive:",
        labels.count("positive")
    )

    print(
        "Negative:",
        labels.count("negative")
    )

    print(
        "Neutral:",
        labels.count("neutral")
    )


    # -----------------------------------------------------
    # Baseline Cross-Validation
    # -----------------------------------------------------

    print(
        "\n============================================"
    )

    print(
        "       Baseline Cross-Validation"
    )

    print(
        "============================================"
    )

    print(
        "Cross-validation scores:",
        baseline_cv_scores
    )

    print(
        "Average cross-validation accuracy:",
        round(
            float(baseline_cv_scores.mean()),
            3
        )
    )


    # -----------------------------------------------------
    # Accuracy Comparison
    # -----------------------------------------------------

    print(
        "\n============================================"
    )

    print(
        "             Accuracy Results"
    )

    print(
        "============================================"
    )

    print(
        "Logistic Regression Accuracy "
        "Before GridSearchCV:",
        round(
            float(baseline_accuracy),
            3
        )
    )

    print(
        "Logistic Regression Accuracy "
        "After GridSearchCV:",
        round(
            float(optimized_accuracy),
            3
        )
    )


    # -----------------------------------------------------
    # Best GridSearchCV Results
    # -----------------------------------------------------

    print(
        "\nBest GridSearchCV Parameters:"
    )

    print(
        grid_search.best_params_
    )

    print(
        "\nBest GridSearchCV "
        "Cross-Validation Accuracy:",
        round(
            float(grid_search.best_score_),
            3
        )
    )


    # -----------------------------------------------------
    # Baseline Classification Report
    # -----------------------------------------------------

    print(
        "\n============================================"
    )

    print(
        "Classification Report Before GridSearchCV"
    )

    print(
        "============================================"
    )

    print(
        classification_report(
            y_test,
            baseline_predictions,
            zero_division=0
        )
    )


    # -----------------------------------------------------
    # Optimized Classification Report
    # -----------------------------------------------------

    print(
        "\n============================================"
    )

    print(
        "Classification Report After GridSearchCV"
    )

    print(
        "============================================"
    )

    print(
        classification_report(
            y_test,
            optimized_predictions,
            zero_division=0
        )
    )


    # -----------------------------------------------------
    # Best Model
    # -----------------------------------------------------

    print(
        "\n============================================"
    )

    print(
        "                Best Model"
    )

    print(
        "============================================"
    )

    print(
        "Best Model:",
        best_model_name
    )

    print(
        "Best Model Accuracy:",
        round(
            float(best_model_accuracy),
            3
        )
    )


    # -----------------------------------------------------
    # Test Sentences
    # -----------------------------------------------------

    test_sentences = [
        "The application works perfectly.",
        "The service is very bad.",
        "The meeting starts tomorrow.",
        "I am extremely happy with the results.",
        "I am disappointed with this product.",
        "The document contains ten pages.",
        "The team completed the task successfully.",
        "The application keeps crashing.",
        "The report was uploaded yesterday."
    ]


    print(
        "\n============================================"
    )

    print(
        "         Sentiment Test Results"
    )

    print(
        "============================================"
    )


    for sentence in test_sentences:

        result = analyze_sentiment(
            sentence
        )

        print(
            "\nText:",
            sentence
        )

        print(
            "Predicted Sentiment:",
            result["result"]["label"]
        )

        print(
            "Confidence:",
            result["confidence"]
        )

        print(
            "-" * 50
        )