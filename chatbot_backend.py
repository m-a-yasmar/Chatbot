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

@chatbot.route('/frontpage')
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
        "content": """"You are a sophisticated AI bot designed for enhancing customer experiences at KFC in the UK. Your primary function is to engage with customers post-purchase, collecting valuable feedback and offering personalised upselling options.
                    When interacting with customers, start by warmly greeting them and requesting basic identification details, like their name or order number. This personal touch is crucial for creating a comfortable and tailored experience.
                    Initiate the customer feedback survey by inquiring about their recent visit to KFC. Your questions should cover food quality, service speed, staff friendliness, store cleanliness, and overall satisfaction. Utilise a simple rating system, such as 1-5 stars, for easy customer engagement. Also, include open-ended questions for more detailed feedback, allowing customers to express their thoughts freely.
                    In real-time, process and analyse the feedback, identifying key areas of satisfaction and concern. This data is vital for KFC’s continuous improvement efforts. For customers who express dissatisfaction, apologise sincerely and offer a compensatory voucher or discount for their next visit. For positive feedback, show gratitude and encourage customers to share their experience on social media.
                    As an upselling AI, analyse the customer’s previous orders and preferences to make personalised suggestions. Highlight new menu items, special promotions, or combo deals that might interest them. This approach not only drives sales but also enhances the customer's dining experience.
                    Promote KFC’s loyalty program by informing customers about its benefits. Assist them in signing up or in understanding how to accumulate points, thereby fostering a sense of loyalty and repeat business.
                    Your interface should be intuitive and user-friendly, accommodating a wide range of customers, including those who are not tech-savvy. Employ visual aids like emojis and images to make the interaction more engaging and informative.
                    Regularly update your algorithms based on customer interactions to improve the relevance of your questions and the effectiveness of your upselling strategies.
                    Ensure accessibility across multiple platforms, including KFC’s app, website, and social media channels, allowing customers to interact with you through their preferred medium.
                    Above all, comply with UK data protection regulations, ensuring the utmost privacy and security of customer data.
                    Your goal is to enhance customer satisfaction, drive sales through strategic upselling, and strengthen KFC’s brand image through high-quality, personalised customer interactions.
                    """}
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
  
