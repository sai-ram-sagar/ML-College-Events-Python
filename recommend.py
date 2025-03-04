from flask import Flask, request, jsonify
import sqlite3
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def get_events():
    with open("events.json", "r", encoding="utf-8") as file:
        events = json.load(file)
    return events

def get_user_history(user_id):
    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()
    cursor.execute("SELECT search_query FROM search_history WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [row[0].lower() for row in rows] if rows else []

def recommend_events(user_id, top_n=4):
    user_history = get_user_history(user_id)
    all_events = get_events()

    if not user_history or not all_events:
        return []

    event_names = [event["event_name"].lower() for event in all_events]
    user_search_text = " ".join(user_history)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([user_search_text] + event_names)

    user_vector = tfidf_matrix[0]
    event_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(user_vector, event_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    recommended_events = [all_events[i] for i in top_indices]

    return recommended_events

@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user_id", type=int)
    if user_id is None:
        return jsonify({"error": "User ID is required"}), 400
    
    recommendations = recommend_events(user_id)
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
