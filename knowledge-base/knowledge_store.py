
import sqlite3

DB_PATH = "smart_assistant.db"

def save_interaction(type, input_data, output_json):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO interactions (type, input_data, output_json)
        VALUES (?, ?, ?)
    ''', (type, input_data, output_json))
    conn.commit()
    conn.close()


def log_sentiment(text, label, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO sentiment_log (text, label, confidence)
        VALUES (?, ?, ?)
    ''', (text, label, confidence))
    conn.commit()
    conn.close()

def log_intent(text, intent, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO intent_log (text, intent, confidence)
        VALUES (?, ?, ?)
    ''', (text, intent, confidence))
    conn.commit()
    conn.close()


def get_recent(n):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT *
        FROM interactions
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (n,))
    results = cursor.fetchall()
    conn.close()
    return results     