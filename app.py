
import sys
import os
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'integration'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'knowledge-base'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nlp-module'))

from assistant_core import process_input
from knowledge_store import get_recent, sentiment_summary, top_keywords_overall
from db_setup import init_db

app = Flask(__name__)
init_db()

@app.route('/assistant/text', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    result = process_input(text)
    return jsonify(result)


@app.route('/assistant/image', methods=['POST'])
def analyze_image():
    data = request.get_json()
    image_path = data.get('image_path', '')
    
    if not image_path:
        return jsonify({"error": "No image path provided"}), 400
    
    return jsonify({
        "type": "face",
        "result": {"detected": False, "count": 0},
        "confidence": 0.0,
        "message": "Image analysis coming soon"
    })


@app.route('/assistant/history', methods=['GET'])
def get_history():
    interactions = get_recent(10)
    result = []
    for interaction in interactions:
        result.append({
            "id": interaction[0],
            "type": interaction[1],
            "input": interaction[2],
            "output": interaction[3],
            "timestamp": interaction[4]
        })
    return jsonify(result)

@app.route('/assistant/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "sentiment_summary": sentiment_summary(),
        "top_keywords": top_keywords_overall()
    })



if __name__ == '__main__':
    app.run(debug=True)    