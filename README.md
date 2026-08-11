# Smart Multi-Modal Assistant

A multi-modal intelligent assistant that combines **Natural Language Processing (NLP)** and **Computer Vision (CV)** through a unified processing pipeline.
The system can analyze text and images, expose its capabilities through a **command-line interface (CLI)** and a **Flask REST API**, and persist interactions and analysis results using **SQLite**.
The project was developed as a team-based software engineering project, with separate modules for NLP, Computer Vision, database management, and system integration.

---

## Key Features

### Text Analysis

* Intent classification, identifying user intent such as questions, commands, complaints, and greetings.
* Sentiment analysis, classifying text as positive, negative, or neutral.
* Keyword extraction, extracting important keywords from user input.
* Extractive summarization.

### Image Analysis

* Face detection using OpenCV.
* Face counting.
* Face visualization through bounding boxes.

### Data and Statistics

* Persistent interaction logging using SQLite.
* Stores text and image analysis results.
* Tracks sentiment and intent predictions.
* Provides interaction history.
* Provides basic statistics, including sentiment distribution and frequent keywords.

### Interfaces

* Command-line interface.
* Flask-based REST API.
* Unified integration layer connecting the NLP, CV, database, and interface components.

---

## Architecture

The system follows a modular pipeline where user requests enter through either the **CLI** or **Flask REST API**. The `Assistant Core` coordinates the appropriate AI module, processes the result, and stores the interaction and analysis results in the SQLite database.

```text
                           ┌──────────────┐
                           │     User     │
                           └──────┬───────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             ┌─────────────┐             ┌─────────────┐
             │     CLI     │             │ Flask REST  │
             │  main.py    │             │   app.py    │
             └──────┬──────┘             └──────┬──────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                       ┌────────────────────┐
                       │   Assistant Core   │
                       │ assistant_core.py  │
                       └─────────┬──────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
             ┌─────────────┐           ┌─────────────┐
             │ NLP Module  │           │  CV Module  │
             │             │           │             │
             │ • Intent    │           │ • Face      │
             │ • Sentiment │           │   Detection │
             │ • Keywords  │           │ • Face Count│
             │ • Summary   │           │             │
             └──────┬──────┘           └──────┬──────┘
                    │                         │
                    └───────────┬─────────────┘
                                ▼
                       ┌─────────────────┐
                       │ Analysis Result │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ SQLite Database │
                       │                 │
                       │ • interactions │
                       │ • sentiment_log│
                       │ • intent_log   │
                       └─────────────────┘
```

### Processing Flow

1. The user submits text or an image through the CLI or Flask API.
2. The request is passed to the Assistant Core.
3. The Assistant Core routes the input to the appropriate NLP or CV module.
4. The selected module performs the required analysis.
5. The analysis result is returned to the Assistant Core.
6. The interaction and relevant analysis results are stored in SQLite.
7. The result is returned to the user through the CLI or API.

---

## Project Structure

```text
smart-assistant-girls-team/
├── README.md
├── requirements.txt
├── app.py                         # Flask REST API
├── main.py                        # CLI entry point
│
├── nlp-module/
│   ├── intent_classifier.py
│   ├── sentiment_analyzer.py
│   ├── keyword_extractor.py
│   ├── summarizer.py
│   ├── README.md
│   └── EVALUATION.md
│
├── cv-module/
│   ├── face_detector.py
│   ├── image_classifier.py
│   ├── models/
│   │   └── haarcascade_frontalface_default.xml
│   ├── README.md
│   ├── EVALUATION.md
│   └── HANDOVER.md
│
├── knowledge-base/
│   ├── db_setup.py
│   └── knowledge_store.py
│
├── integration/
│   ├── assistant_core.py
│   └── INTERFACE.md
│
├── interface/
│   └── cli_interface.py
│
├── docs/
│   ├── API.md
│   └── db_schema.md
│
└── tests/
    ├── test_nlp.py
    ├── test_cv.py
    ├── test_integration.py
    └── tests_images/
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ShahadAi2002/smart-assistant-girls-team.git
cd smart-assistant-girls-team
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### CLI

Run the command-line interface:

```bash
python main.py
```

### REST API

Start the Flask API:

```bash
python app.py
```

The API will be available at:

```text
http://127.0.0.1:5000
```

---

## API Examples

### Analyze Text

```bash
curl -X POST http://127.0.0.1:5000/assistant/text \
  -H "Content-Type: application/json" \
  -d '{"text": "How do I reset my password?"}'
```

Example response:

```json
{
  "intent":    {"type": "intent",    "result": {"label": "question"}, "confidence": 0.33},
  "sentiment": {"type": "sentiment", "result": {"label": "neutral"},  "confidence": 0.42},
  "keywords":  {"type": "keywords",  "result": {"keywords": ["reset", "password"]}, "confidence": 0.71}
}
```

### Analyze Image

```bash
curl -X POST http://127.0.0.1:5000/assistant/image \
  -H "Content-Type: application/json" \
  -d '{"image_path": "images/photo.jpg"}'
```

Example response:

```json
{
  "type": "face",
  "result": {
    "detected": true,
    "count": 2
  },
  "confidence": 0.91
}
```

### Get Interaction History

Retrieve the latest interactions:

```bash
curl http://127.0.0.1:5000/assistant/history
```

### Get Statistics

Retrieve system statistics:

```bash
curl http://127.0.0.1:5000/assistant/stats
```

For full API documentation, see [`docs/API.md`](docs/API.md).

---

## Database Design

### interactions

| Column      | Type       | Description                               |
| ----------- | ---------- | ----------------------------------------- |
| id          | INTEGER PK | Primary key, auto-incremented             |
| type        | TEXT       | Input type (`text` or `image`)            |
| input_data  | TEXT       | The original input provided               |
| output_json | TEXT       | The full result returned by the assistant |
| timestamp   | DATETIME   | When the interaction occurred             |

### sentiment_log

| Column     | Type       | Description                                         |
| ---------- | ---------- | --------------------------------------------------- |
| id         | INTEGER PK | Primary key, auto-incremented                       |
| text       | TEXT       | The input text analyzed                             |
| label      | TEXT       | Sentiment label (`positive`, `negative`, `neutral`) |
| confidence | FLOAT      | Confidence score between 0.0 and 1.0                |
| timestamp  | DATETIME   | When the sentiment was logged                       |

### intent_log

| Column     | Type       | Description                                                      |
| ---------- | ---------- | ---------------------------------------------------------------- |
| id         | INTEGER PK | Primary key, auto-incremented                                    |
| text       | TEXT       | The input text analyzed                                          |
| intent     | TEXT       | Detected intent (`question`, `command`, `complaint`, `greeting`) |
| confidence | FLOAT      | Confidence score between 0.0 and 1.0                             |
| timestamp  | DATETIME   | When the intent was logged                                       |

---

## Testing

Run the complete test suite with:

```bash
pytest tests/
```

The project includes tests covering:

* NLP functionality
* Computer Vision functionality
* Integration between modules
---

## Team and Responsibilities

| Member | Role                         | Responsibilities                                                                     |
| ------ | ---------------------------- | ------------------------------------------------------------------------------------ |
| Shahad | Integration + Database + API | `assistant_core.py`, SQLite database, Flask API, integration tests                   |
| Atheer | NLP Module + CLI             | Intent classifier, sentiment analyzer, keyword extractor, summarizer, CLI, NLP tests |
| Rahaf  | CV Module                    | Face detector, face visualization, CV functionality, and CV tests                    |


---

## Documentation

Additional documentation is available in:

* [`docs/API.md`](docs/API.md) for API endpoints and usage
* [`docs/db_schema.md`](docs/db_schema.md) for database schema
* [`nlp-module/README.md`](nlp-module/README.md) for NLP module documentation
* [`nlp-module/EVALUATION.md`](nlp-module/EVALUATION.md) for NLP evaluation
* [`cv-module/README.md`](cv-module/README.md) for CV module documentation
* [`cv-module/EVALUATION.md`](cv-module/EVALUATION.md) for CV evaluation
* [`integration/INTERFACE.md`](integration/INTERFACE.md) for integration interface
