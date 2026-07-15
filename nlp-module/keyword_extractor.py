from sklearn.feature_extraction.text import TfidfVectorizer


# Reference texts used to compare the input text
REFERENCE_TEXTS = [
    "Artificial intelligence helps computers perform intelligent tasks",
    "Machine learning allows systems to learn from data",
    "Natural language processing helps computers understand human language",
    "Data science uses statistics and programming to analyze data",
    "Deep learning uses neural networks to solve complex problems"
]


def extract_keywords(text, top_n=5):
    """
    Extract the most important keywords from one text using TF-IDF.

    Parameters:
        text (str): The input text.
        top_n (int): Number of keywords to return.

    Returns:
        dict: Unified output format.
    """

    # Check if the text is valid
    if not isinstance(text, str) or not text.strip():
        return {
            "type": "keywords",
            "result": {
                "keywords": []
            },
            "confidence": 0.0
        }

    # Check if top_n is valid
    if not isinstance(top_n, int) or top_n <= 0:
        top_n = 5

    # Add the input text to the reference texts
    documents = REFERENCE_TEXTS + [text.strip()]

    try:
        # Create the TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 1)
        )

        # Convert all texts into TF-IDF values
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Get the TF-IDF scores for the input text
        text_scores = tfidf_matrix[-1].toarray().flatten()

        # Get all words found by the vectorizer
        feature_names = vectorizer.get_feature_names_out()

        # Connect every word with its TF-IDF score
        keyword_scores = []

        for word, score in zip(feature_names, text_scores):
            if score > 0:
                keyword_scores.append((word, float(score)))

        # Sort words from highest score to lowest score
        keyword_scores.sort(
            key=lambda item: item[1],
            reverse=True
        )

        # Select only the top keywords
        top_keyword_scores = keyword_scores[:top_n]

        keywords = []

        for word, score in top_keyword_scores:
            keywords.append(word)

        # Calculate confidence using the average TF-IDF score
        if top_keyword_scores:
            total_score = sum(
                score for word, score in top_keyword_scores
            )

            confidence = total_score / len(top_keyword_scores)

        else:
            confidence = 0.0

        # Return the unified output format
        return {
            "type": "keywords",
            "result": {
                "keywords": keywords
            },
            "confidence": round(float(confidence), 2)
        }

    except ValueError:
        return {
            "type": "keywords",
            "result": {
                "keywords": []
            },
            "confidence": 0.0
        }


def batch_extract(texts: list, top_n=5):
    """
    Extract keywords from a list of texts.

    Parameters:
        texts (list): A list of text strings.
        top_n (int): Number of keywords to extract from each text.

    Returns:
        list: A list of keyword extraction results.
    """

    # Check if texts is a valid list
    if not isinstance(texts, list):
        return []

    results = []

    # Extract keywords from each text
    for text in texts:
        result = extract_keywords(text, top_n)
        results.append(result)

    return results


# Test the functions
if __name__ == "__main__":

    sample_texts = [
        "Machine learning and artificial intelligence are used to analyze data",
        "Natural language processing helps computers understand human language",
        "Deep learning uses neural networks to solve complex problems"
    ]

    results = batch_extract(sample_texts, top_n=5)

    for index, result in enumerate(results, start=1):
        print(f"Text {index}:")
        print(result)
        print()