"""
Text summarization module.

This module provides simple extractive text summarization using TF-IDF.
It selects the highest-scoring sentences while preserving their
original order in the text.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer


def summarize(text, n_sentences=2):
    """
    Create a simple extractive summary using TF-IDF.

    Args:
        text (str): The input text.
        n_sentences (int): Number of sentences to include.

    Returns:
        str: The summarized text.
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

    # Split the text into sentences
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    # Return the full text if the requested number
    # is greater than or equal to the sentence count
    if len(sentences) <= n_sentences:
        return " ".join(sentences)

    # Convert sentences into TF-IDF vectors
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        sentences
    )

    # Calculate a TF-IDF score for each sentence
    sentence_scores = tfidf_matrix.sum(
        axis=1
    ).A1

    # Select the highest-scoring sentences
    top_indices = sentence_scores.argsort()[
        -n_sentences:
    ]

    # Restore the original sentence order
    top_indices = sorted(top_indices)

    summary_sentences = [
        sentences[index]
        for index in top_indices
    ]

    return " ".join(summary_sentences)


def summarize_text(text, n_sentences=2):
    """
    Summarize text using the main summarize() function.

    This wrapper is kept for compatibility with cli_interface.py.

    Args:
        text (str): The input text.
        n_sentences (int): Number of sentences to include.

    Returns:
        str: The summarized text.
    """

    return summarize(
        text,
        n_sentences=n_sentences
    )


if __name__ == "__main__":
    sample_text = (
        "Artificial intelligence is growing rapidly. "
        "It is used in healthcare and education. "
        "Machine learning is an important part of AI. "
        "Many companies use AI to improve their services."
    )

    result = summarize(
        sample_text,
        n_sentences=2
    )

    print("Summary:")
    print(result)