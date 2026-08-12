# NLP Module

This module contains the main Natural Language Processing functions used in the Smart Assistant project.

## Intent Classification

### Function

```python
classify_intent(text)
```

This function identifies the intent of the input text.

Possible labels:

- question
- command
- complaint
- greeting

Example:

```python
result = classify_intent("How do I open the file?")
print(result)
```

Example output:

```python
{
    "type": "intent",
    "result": {
        "label": "question"
    },
    "confidence": 0.80
}
```

## Sentiment Analysis

### Function

```python
analyze_sentiment(text)
```
This function analyzes the sentiment of the input text.

Possible labels:

- positive
- negative
- neutral

Example:

```python
result = analyze_sentiment(
    "I am satisfied with the result"
)

print(result)
```

Example output:

```python
{
    "type": "sentiment",
    "result": {
        "label": "positive"
    },
    "confidence": 0.75
}
```

## Keyword Extraction

### Function

```python
extract_keywords(text)
```

This function extracts the most important keywords from the input text.

Example:

```python
result = extract_keywords(
    "Artificial intelligence improves data analysis"
)

print(result)
```

Example output:

```python
{
    "type": "keywords",
    "result": {
        "keywords": [
            "artificial",
            "intelligence",
            "improves",
            "data",
            "analysis"
        ]
    },
    "confidence": 1.0
}
```

## Unified Output Format

The three NLP functions use the following unified output format:

```python
{
    "type": "...",
    "result": {
        ...
    },
    "confidence": 0.0
}
```

## CLI Interface

The NLP functions are connected to:

```text
interface/cli_interface.py
```

The interface displays:

1. Intent classification
2. Sentiment analysis
3. Keyword extraction
4. Temporary text summary

The interface also handles:

- Empty text
- Invalid menu choices
- Invalid function results
- Unexpected errors