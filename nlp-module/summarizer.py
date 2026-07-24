import re

from sklearn.feature_extraction.text import TfidfVectorizer


def summarize(text, n_sentences=2):
    """
    Create a simple extractive summary using TF-IDF.

    The function:
    1. Splits the text into sentences.
    2. Calculates a TF-IDF score for each sentence.
    3. Selects the most important sentences.
    4. Returns them in their original order.
    """

    if not isinstance(text, str):
        raise TypeError("The input text must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("The input text cannot be empty.")

    if not isinstance(n_sentences, int):
        raise TypeError("n_sentences must be an integer.")

    if n_sentences <= 0:
        raise ValueError("n_sentences must be greater than zero.")

    # Split text into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Remove empty sentences
    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    # If the text has fewer sentences than requested
    if len(sentences) <= n_sentences:
        return " ".join(sentences)

    # Convert sentences into TF-IDF vectors
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Calculate a score for every sentence
    sentence_scores = tfidf_matrix.sum(axis=1)

    scores = [
        float(sentence_scores[index, 0])
        for index in range(len(sentences))
    ]

    # Select the indexes of the highest-scoring sentences
    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )[:n_sentences]

    # Return selected sentences in their original order
    ranked_indexes.sort()

    summary_sentences = [
        sentences[index]
        for index in ranked_indexes
    ]

    return " ".join(summary_sentences)


if __name__ == "__main__":
    sample_text = (
        "Artificial intelligence is changing many industries. "
        "It helps automate repetitive tasks. "
        "Machine learning allows systems to learn from data. "
        "Many companies use artificial intelligence to improve services."
    )

    result = summarize(
        sample_text,
        n_sentences=2
    )

    print("Original text:")
    print(sample_text)

    print("\nSummary:")
    print(result)