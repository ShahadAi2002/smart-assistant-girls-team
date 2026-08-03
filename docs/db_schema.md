# Database Schema

## 1. interactions
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| type | TEXT | Type of interaction (text/image) |
| input_data | TEXT | The input provided by the user |
| output_json | TEXT | The output returned by the assistant |
| timestamp | DATETIME | When the interaction happened |

## 2. sentiment_log
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| text | TEXT | The input text |
| label | TEXT | Sentiment label  |
| confidence | FLOAT | Confidence score of the prediction |
| timestamp | DATETIME | When the sentiment was logged |

## 3. intent_log
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| text | TEXT | The input text |
| intent | TEXT | The detected intent |
| confidence | FLOAT | Confidence score of the prediction |
| timestamp | DATETIME | When the intent was logged |
