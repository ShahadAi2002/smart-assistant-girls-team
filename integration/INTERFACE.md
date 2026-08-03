# Unified Interface Documentation

## Standard Output Format
Every module must return a dictionary in the following format:

```python
{
    "type": str,        # the type of module (intent, sentiment, keywords, face)
    "result": dict,     # the actual output of the module
    "confidence": float # confidence score between 0.0 and 1.0
}
```

## Modules & Examples

### 1. classify_intent(text)
- **Input:** `text` (str)
- **Output:**
```python
{"type": "intent", "result": {"intent": "weather_query"}, "confidence": 0.88}
```

### 2. analyze_sentiment(text)
- **Input:** `text` (str)
- **Output:**
```python
{"type": "sentiment", "result": {"label": "positive"}, "confidence": 0.95}
```

### 3. extract_keywords(text)
- **Input:** `text` (str)
- **Output:**
```python
{"type": "keywords", "result": {"keywords": ["weather", "today"]}, "confidence": 0.80}
```

### 4. detect_faces(image_path)
- **Input:** `image_path` (str)
- **Output:**
```python
{"type": "face", "result": {"detected": True, "count": 1}, "confidence": 0.91}
``` 