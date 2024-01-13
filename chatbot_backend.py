import os  # for environment variables
import json  # for JSON handling
import numpy as np  # for numerical operations
from sklearn.feature_extraction.text import TfidfVectorizer  # for TF-IDF
from sklearn.metrics.pairwise import cosine_similarity  # for cosine similarity
from flask import Flask, request, jsonify, render_template, make_response  # for Flask
from flask_session import Session  # Import for Flask-Session
import redis  # Import for Redis
import requests  # for HTTP requests
from flask import send_from_directory # To help insert image
from flask import session #for keeping history
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
#from uuid import uuid4

import psycopg2
import re
import logging
import tempfile
from flask import send_file
from flask_cors import CORS # for CORS
import uuid
from flask import Response
from datetime import timedelta



chatbot = Flask(__name__)
chatbot.secret_key = 'michaelramsay_secret_redis'
CORS(chatbot, supports_credentials=True)

def init_db():
    conn = psycopg2.connect(os.environ['DATABASE_URL'], sslmode='require')
    cur = conn.cursor()

    # Drop the existing table if it exists
    cur.execute("DROP TABLE IF EXISTS chatbot_schema.conversations;")

    # Create new schema if it doesn't exist
    cur.execute("CREATE SCHEMA IF NOT EXISTS chatbot_schema;")

    # Create table in the new schema with a unique constraint on user_id
    cur.execute("""
        CREATE TABLE chatbot_schema.conversations (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(50) NOT NULL,
            conversation_history TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (user_id)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    
init_db()

def generate_unique_id():
    return str(uuid.uuid4())

@chatbot.route('/', methods=['GET'])
def home():
    return render_template('frontpage.html')

@chatbot.route('/image/<path:filename>')
def serve_image(filename):
    return send_from_directory('image', filename)

@chatbot.route('/home')
def frontpage():
    return render_template('frontpage.html')

@chatbot.route('/contact')
def contact():
    return render_template('contact.html')

@chatbot.route('/services')
def services():
    return render_template('services.html')

@chatbot.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_id = data.get('user_id')
    query = data.get('query')
    tokens = query.split()
    max_tokens = 50

    if not user_id or not query:
        return jsonify({"answer": "Missing user_id or query in the request"}), 400

    if len(tokens) > max_tokens:
        return jsonify({"answer": "Your query is too long. Please limit it to 50 words or less."})
    custom_prompt = {
        "role": "system",
        "content": """"Task: Assist in creating a personalized travel plan for a customer using an AI-powered chatbot. The chatbot should engage in a conversation that follows these steps:
                        1. Greet the customer and gather basic travel preferences.
                           - Ask about preferred destinations, travel dates, budget, and interests.
                        2. Suggest destinations and itineraries based on the initial preferences.
                           - Provide options and ask follow-up questions to refine choices (activities, accommodation, attractions).
                        3. Assist in building an interactive travel itinerary.
                           - Present choices for flights, hotels, activities, and dining for each destination. Use engaging language and, if possible, include links or visuals.
                        4. Seek feedback on the proposed travel plan and make adjustments.
                           - Ask for the customer's thoughts on the draft itinerary and offer to modify it based on their feedback.
                        5. Integrate real-time data for up-to-date information.
                           - Provide current information on pricing, availability, and special offers.
                        6. Introduce gamification elements.
                           - Award points for completing steps in the planning process, which can be redeemed for discounts or extras. Include mini-games or quizzes related to travel planning.
                        7. Guide through the booking process.
                           - Once the itinerary is finalized, assist with the booking of flights, hotels, and activities. Upsell or cross-sell additional services like travel insurance.
                        8. Engage post-booking.
                           - Provide travel tips, reminders, and updates about the upcoming trip. Collect feedback post-trip to improve future services.
                        Remember to maintain a friendly and helpful tone throughout the conversation. Ensure the chatbot responses are user-focused, providing a seamless and enjoyable travel planning experience.
                        **Sample Interaction:**
                        Chatbot: "Hello! I'm excited to help you plan your perfect trip. To start, could you tell me which destinations you're interested in, your preferred travel dates, budget, and any specific interests or activities you enjoy?"
                        [Customer provides their preferences]
                        Chatbot: "Great choices! Based on your preferences, I recommend [Destination]. It's wonderful for [Activities] and fits well within your budget. For accommodations, do you prefer hotels, hostels, or private rentals?"
                        [Continue the conversation following the steps outlined above.]"""}
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'], sslmode='require')
        cur = conn.cursor()

        # Check if there's an existing conversation
        cur.execute("SELECT conversation_history FROM chatbot_schema.conversations WHERE user_id = %s", (user_id,))
        row = cur.fetchone()

        if row:
            # If conversation exists, load the history
            conversation_history = json.loads(row[0])
        else:
            # If no conversation, start with custom prompt
            conversation_history = [custom_prompt]

        conversation_history.append({"role": "user", "content": query})

        api_endpoint = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4-1106-preview",
            "messages": conversation_history,
            "frequency_penalty": 1.0,
            "presence_penalty": -0.5
        }
        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content']
            forbidden_phrases = ["I am a model trained", "As an AI model", "My training data includes", "ChatGPT", "OpenAI"]
            for phrase in forbidden_phrases:
                answer = answer.replace(phrase, "")
            conversation_history.append({"role": "assistant", "content": answer})
        else:
            answer = "I'm sorry, I couldn't understand the question."

        # Update or insert the conversation history
        cur.execute("INSERT INTO chatbot_schema.conversations (user_id, conversation_history) VALUES (%s, %s) ON CONFLICT (user_id) DO UPDATE SET conversation_history = %s", (user_id, json.dumps(conversation_history), json.dumps(conversation_history)))
        conn.commit()
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"answer": "An error occurred. Please try again."}), 500

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    chatbot.run(host='0.0.0.0', port=port)
  
