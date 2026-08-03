
# Smart Assistant API Reference

---

### Analyze Text

Analyze input text and return intent, sentiment, and keyword extraction results.

#### HTTP Method and URL
`POST /assistant/text`

#### Example Request
```json
{"text": "How do I open the file?"}
```

| Element | Type   | Description               | Required? |
|---------|--------|---------------------------|-----------|
| text    | String | The input text to analyze | Required  |

#### Example Response
```json
{
  "intent":    {"type": "intent",    "result": {"label": "question"}, "confidence": 0.33},
  "sentiment": {"type": "sentiment", "result": {"label": "neutral"},  "confidence": 0.42},
  "keywords":  {"type": "keywords",  "result": {"keywords": ["file", "open"]}, "confidence": 0.71}
}
```

#### Error and Status Codes

| Code | Message          | Meaning                        |
|------|------------------|--------------------------------|
| 200  | OK               | Request was successful         |
| 400  | No text provided | Request body is missing `text` |

---

### Analyze Image

Submit an image path for face detection analysis.

#### HTTP Method and URL
`POST /assistant/image`

#### Example Request
```json
{"image_path": "images/photo.jpg"}
```

| Element    | Type   | Description           | Required? |
|------------|--------|-----------------------|-----------|
| image_path | String | Path to the image file | Required  |

#### Example Response
```json
{
  "type": "face",
  "result": {"detected": true, "count": 2},
  "confidence": 0.91
}
```

#### Error and Status Codes

| Code | Message                | Meaning                              |
|------|------------------------|--------------------------------------|
| 200  | OK                     | Request was successful               |
| 400  | No image path provided | Request body is missing `image_path` |

---

### Get Interaction History

Retrieve the last 10 interactions stored in the database.

#### HTTP Method and URL
`GET /assistant/history`

#### Parameters
None

#### Example Response
```json
[
  {
    "id": 20,
    "type": "text",
    "input": "I am very happy today",
    "output": "{...}",
    "timestamp": "2026-07-30 19:00:56"
  }
]
```

#### Error and Status Codes

| Code | Message | Meaning                |
|------|---------|------------------------|
| 200  | OK      | Request was successful |

---

### Get Statistics

Retrieve sentiment distribution and top keywords across all interactions.

#### HTTP Method and URL
`GET /assistant/stats`

#### Parameters
None

#### Example Response
```json
{
  "sentiment_summary": {"positive": 7, "negative": 2, "neutral": 6},
  "top_keywords": ["file", "open", "amazing", "service", "hello"]
}
```

#### Error and Status Codes

| Code | Message | Meaning                |
|------|---------|------------------------|
| 200  | OK      | Request was successful | 