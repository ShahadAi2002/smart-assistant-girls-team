import sqlite3
import ast

DB_PATH = "smart_assistant.db"


def initialize_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            input_data TEXT NOT NULL,
            output_json TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS intent_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            intent TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


initialize_database()


def save_interaction(type, input_data, output_json):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO interactions (type, input_data, output_json)
        VALUES (?, ?, ?)
    """, (type, input_data, output_json))

    conn.commit()
    conn.close()


def log_sentiment(text, label, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO sentiment_log (text, label, confidence)
        VALUES (?, ?, ?)
    """, (text, label, confidence))

    conn.commit()
    conn.close()


def log_intent(text, intent, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO intent_log (text, intent, confidence)
        VALUES (?, ?, ?)
    """, (text, intent, confidence))

    conn.commit()
    conn.close()


def get_recent(n):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM interactions
        ORDER BY timestamp DESC
        LIMIT ?
    """, (n,))

    results = cursor.fetchall()
    conn.close()

    return results


def sentiment_summary():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT label, COUNT(*)
        FROM sentiment_log
        GROUP BY label
    """)

    results = cursor.fetchall()
    conn.close()

    summary = {
        "positive": 0,
        "negative": 0,
        "neutral": 0
    }

    for label, count in results:
        summary[label] = count

    return summary


def top_keywords_overall(n=10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT output_json
        FROM interactions
        WHERE type = 'text'
    """)

    results = cursor.fetchall()
    conn.close()

    keyword_counts = {}

    for (output_json,) in results:
        try:
            data = ast.literal_eval(output_json)

            keywords = (
                data.get("keywords", {})
                .get("result", {})
                .get("keywords", [])
            )

            for keyword in keywords:
                keyword_counts[keyword] = (
                    keyword_counts.get(keyword, 0) + 1
                )

        except (ValueError, SyntaxError, AttributeError):
            continue

    sorted_keywords = sorted(
        keyword_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        keyword
        for keyword, count in sorted_keywords[:n]
    ]