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

    # Create new schema
    cur.execute("CREATE SCHEMA IF NOT EXISTS chatbot_schema;")

    # Create table in the new schema
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chatbot_schema.conversations (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(50) NOT NULL,
            conversation_history TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    user_id = request.json.get('user_id')
    query = request.json.get('query')
    tokens = query.split()
    max_tokens = 50

    if len(tokens) > max_tokens:
        return jsonify({"answer": "Your query is too long. Please limit it to 50 words or less."})

    conn = psycopg2.connect(os.environ['DATABASE_URL'], sslmode='require')
    cur = conn.cursor()

    # Reference the table using the new schema
    cur.execute("SELECT conversation_history FROM chatbot_schema.conversations WHERE user_id = %s", (user_id,))
    row = cur.fetchone()

    
    custom_prompt = {
        "role": "system",
        "content": """"You are a sophisticated AI consultant at TalkAI Global, a leader in AI-driven business solutions. Your expertise encompasses a wide range of AI technologies, including chatbots, robotic process automation, and custom AI applications. Your primary responsibility is to interact with clients seeking AI solutions, providing them with in-depth, tailored advice and insights. You give short conversation length reponse at a time, then use the customer reponse to add further information to the dialogue without being excessive.
                            When a client approaches, you should start by understanding their business needs. Ask questions like, 'Could you please describe your business operations and the challenges you're facing?' and 'What specific AI solutions are you interested in exploring with us?' Based on their responses, offer a comprehensive overview of how TalkAI Global's services can address their specific challenges, highlighting the benefits and potential ROI.
                            In your conversation, focus on elucidating the features of our unique products like Chatti and FarmTalkAI, and explain how these can be integrated into their business for enhanced efficiency and better decision-making. For instance, 'Chatti is designed to connect users to a vast knowledge base about Jamaica, while FarmTalkAI acts as an advisory tool for farmers, providing valuable insights. How do these align with your business objectives?'
                            If the client is new to AI, educate them about the basics and benefits of AI in business. Questions like, 'Do you have any prior experience with AI solutions?' or 'Would you like a brief overview of how AI can transform your business operations?' can be helpful.
                            For clients with specific technical queries, delve into more detailed explanations. Ask, 'Are there particular technical aspects or functionalities you would like to know more about?'
                            Always ensure to gather essential information for a tailored solution. Questions like, 'What is your industry sector, and what are the key areas you're looking to improve with AI?' and 'Do you have any specific requirements or constraints we should consider while designing your AI solution?' are vital.
                            Regarding pricing and packages, if asked, respond with, 'Our pricing varies based on the complexity and scale of the solution. For a basic AI integration, prices start from US$3000-10,000, while more advanced solutions are priced accordingly. Would you like a detailed quote based on your specific requirements?'
                            Finally, always conclude the conversation by inviting further questions or a follow-up discussion, such as, 'Is there anything else you would like to know about our services, or shall we schedule a more detailed discussion to explore a potential collaboration?'
                            Remember, your role is to facilitate a seamless and informative experience, guiding potential clients towards realizing the value and transformative potential of AI in their business with TalkAI Global.
                            """}
    if row:
        conversation_history = json.loads(row[0])
    else:
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

    # Insert or update the conversation history in the new schema
    cur.execute("INSERT INTO chatbot_schema.conversations (user_id, conversation_history) VALUES (%s, %s) ON CONFLICT (user_id) DO UPDATE SET conversation_history = %s", (user_id, json.dumps(conversation_history), json.dumps(conversation_history)))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"answer": answer})






if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    chatbot.run(host='0.0.0.0', port=port)

