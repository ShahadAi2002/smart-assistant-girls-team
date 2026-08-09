# Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


def combine_parts(first_parts, second_parts):
    """
    Combine two lists to create unique sentences.

    Example:
        10 first parts × 10 second parts = 100 sentences.
    """
    return [
        f"{first_part} {second_part}".strip()
        for first_part in first_parts
        for second_part in second_parts
    ]


# =========================================================
# Question Dataset: 100 sentences
# =========================================================

question_starters = [
    "Can you explain",
    "Could you tell me",
    "Do you know",
    "Would you clarify",
    "Can you check",
    "Is it possible to know",
    "Could you help me understand",
    "Would you tell me",
    "Can you confirm",
    "Do you understand"
]

question_topics = [
    "why the application is not responding?",
    "how I can open the saved file?",
    "when the next meeting will begin?",
    "where the system stores previous results?",
    "whether the latest update was installed?",
    "how the model chooses its prediction?",
    "why the confidence score is so low?",
    "what caused the program to close?",
    "whether the database is connected?",
    "how I can recover a deleted message?"
]

questions = combine_parts(
    question_starters,
    question_topics
)


# =========================================================
# Command Dataset: 100 sentences
# =========================================================

command_starters = [
    "Please",
    "Could you",
    "I need you to",
    "Make sure you",
    "Try to",
    "Go ahead and",
    "Do not forget to",
    "You should",
    "Kindly",
    "Before continuing,"
]

command_actions = [
    "open the latest saved file.",
    "run the program again.",
    "display the analysis results.",
    "save the current changes.",
    "check the database connection.",
    "update the training dataset.",
    "generate a summary of this text.",
    "show the most important keywords.",
    "restart the application.",
    "verify that all tests pass."
]

commands = combine_parts(
    command_starters,
    command_actions
)


# =========================================================
# Complaint Dataset: 100 sentences
# =========================================================

complaint_starters = [
    "I am disappointed because",
    "I cannot understand why",
    "It is frustrating that",
    "I am having trouble because",
    "The problem is that",
    "I expected better, but",
    "I followed the instructions, yet",
    "I keep noticing that",
    "I am unhappy because",
    "Something is wrong because"
]

complaint_issues = [
    "the application keeps crashing.",
    "the results are still incorrect.",
    "the file refuses to upload.",
    "the program takes too long to respond.",
    "the system does not save my changes.",
    "the model gives the wrong prediction.",
    "the connection disconnects repeatedly.",
    "the interface is difficult to use.",
    "the same error appears every time.",
    "the application closes without warning."
]

complaints = combine_parts(
    complaint_starters,
    complaint_issues
)


# =========================================================
# Greeting Dataset: 100 sentences
# =========================================================

greeting_starters = [
    "Hello,",
    "Hi,",
    "Hey,",
    "Good morning,",
    "Good afternoon,",
    "Good evening,",
    "Greetings,",
    "Welcome,",
    "Hello there,",
    "Hi everyone."
]

greeting_followups = [
    "I hope you are doing well.",
    "It is nice to see you today.",
    "How has your day been?",
    "I am glad to meet you.",
    "I hope everything is going well.",
    "It is good to hear from you.",
    "How are things going?",
    "I hope you are having a good day.",
    "It is a pleasure to speak with you.",
    "Welcome back to the application."
]

greetings = combine_parts(
    greeting_starters,
    greeting_followups
)


# =========================================================
# Create Complete Dataset
# =========================================================

intent_data = {
    "question": questions,
    "command": commands,
    "complaint": complaints,
    "greeting": greetings
}

texts = []
labels = []

for intent_label, intent_sentences in intent_data.items():
    texts.extend(intent_sentences)
    labels.extend(
        [intent_label] * len(intent_sentences)
    )


# =========================================================
# Verify Dataset Size and Balance
# =========================================================

assert len(questions) == 100
assert len(commands) == 100
assert len(complaints) == 100
assert len(greetings) == 100

assert len(texts) == 400
assert len(labels) == 400

assert labels.count("question") == 100
assert labels.count("command") == 100
assert labels.count("complaint") == 100
assert labels.count("greeting") == 100


# =========================================================
# Split RAW Text Before TF-IDF
# This prevents data leakage.
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)


# =========================================================
# Model 1: Naive Bayes Pipeline
# =========================================================

naive_bayes_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True
        )
    ),
    (
        "classifier",
        MultinomialNB()
    )
])

naive_bayes_model.fit(
    X_train,
    y_train
)

nb_predictions = naive_bayes_model.predict(
    X_test
)

nb_accuracy = accuracy_score(
    y_test,
    nb_predictions
)


# =========================================================
# Model 2: Baseline Logistic Regression
# =========================================================

baseline_logistic_model = Pipeline([
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
            max_iter=2000,
            random_state=42
        )
    )
])

baseline_logistic_model.fit(
    X_train,
    y_train
)

baseline_predictions = baseline_logistic_model.predict(
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

param_grid = {
    "tfidf__ngram_range": [
        (1, 1),
        (1, 2)
    ],

    "classifier__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],

    "classifier__solver": [
        "lbfgs"
    ],

    "classifier__class_weight": [
        None,
        "balanced"
    ]
}


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
# Optimized Logistic Regression
# =========================================================

optimized_logistic_model = grid_search.best_estimator_

optimized_predictions = optimized_logistic_model.predict(
    X_test
)

optimized_accuracy = accuracy_score(
    y_test,
    optimized_predictions
)


# =========================================================
# Compare Models
# =========================================================

model_results = {
    "Naive Bayes": {
        "model": naive_bayes_model,
        "accuracy": nb_accuracy
    },

    "Baseline Logistic Regression": {
        "model": baseline_logistic_model,
        "accuracy": baseline_accuracy
    },

    "Optimized Logistic Regression": {
        "model": optimized_logistic_model,
        "accuracy": optimized_accuracy
    }
}


best_accuracy = max(
    result["accuracy"]
    for result in model_results.values()
)


best_model_names = [
    model_name
    for model_name, result in model_results.items()
    if result["accuracy"] == best_accuracy
]


best_model_name = best_model_names[0]

best_model = model_results[
    best_model_name
]["model"]

best_model_accuracy = model_results[
    best_model_name
]["accuracy"]


# =========================================================
# Intent Classification Function
# =========================================================

def classify_intent(text):
    """
    Classify the intent of the given text.
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

    predicted_intent = best_model.predict(
        [text]
    )[0]

    probabilities = best_model.predict_proba(
        [text]
    )[0]

    confidence = probabilities.max()

    return {
        "type": "intent",

        "result": {
            "label": str(
                predicted_intent
            )
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
        "        Intent Model Evaluation"
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
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )


    training_tfidf = naive_bayes_model.named_steps[
        "tfidf"
    ].transform(X_train)


    print(
        "Training TF-IDF shape:",
        training_tfidf.shape
    )

    print(
        "Classes:",
        sorted(set(labels))
    )


    print(
        "\nSentences in each class:"
    )

    print(
        "Question:",
        labels.count("question")
    )

    print(
        "Command:",
        labels.count("command")
    )

    print(
        "Complaint:",
        labels.count("complaint")
    )

    print(
        "Greeting:",
        labels.count("greeting")
    )


    # =====================================================
    # Accuracy Results
    # =====================================================

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
        "Naive Bayes Accuracy:",
        round(
            float(nb_accuracy),
            3
        )
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


    # =====================================================
    # GridSearchCV Results
    # =====================================================

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


    # =====================================================
    # Naive Bayes Report
    # =====================================================

    print(
        "\n============================================"
    )

    print(
        "Classification Report for Naive Bayes"
    )

    print(
        "============================================"
    )


    print(
        classification_report(
            y_test,
            nb_predictions,
            zero_division=0
        )
    )


    # =====================================================
    # Baseline Logistic Regression Report
    # =====================================================

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


    # =====================================================
    # Optimized Logistic Regression Report
    # =====================================================

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


    # =====================================================
    # Best Model
    # =====================================================

    print(
        "\n============================================"
    )

    print(
        "                Best Model"
    )

    print(
        "============================================"
    )


    if len(best_model_names) > 1:

        print(
            "Models tied for best accuracy:",
            ", ".join(best_model_names)
        )


    print(
        "Selected Model:",
        best_model_name
    )


    print(
        "Best Model Accuracy:",
        round(
            float(best_model_accuracy),
            3
        )
    )


    # =====================================================
    # Test Sentences
    # =====================================================

    sample_sentences = [

        "Could you explain why the program stopped?",

        "Please open the latest report.",

        "I am disappointed because the application is slow.",

        "Good morning, I hope you are doing well.",

        "Can you tell me where the file is stored?",

        "Make sure you save all the changes.",

        "The same error keeps appearing every time.",

        "Hello, it is nice to meet you."
    ]


    print(
        "\n============================================"
    )

    print(
        "           Intent Test Results"
    )

    print(
        "============================================"
    )


    for sentence in sample_sentences:

        result = classify_intent(
            sentence
        )


        print(
            "\nText:",
            sentence
        )


        print(
            "Predicted Intent:",
            result["result"]["label"]
        )


        print(
            "Confidence:",
            result["confidence"]
        )


        print(
            "-" * 50
        )