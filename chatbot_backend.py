import os  # for environment variables
import json  # for JSON handling
import numpy as np  # for numerical operations
from sklearn.feature_extraction.text import TfidfVectorizer  # for TF-IDF
from sklearn.metrics.pairwise import cosine_similarity  # for cosine similarity
from flask import Flask, request, jsonify, render_template, make_response  # for Flask
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

    
def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = psycopg2.connect(os.environ['DATABASE_URL'], sslmode='require')
    cur = conn.cursor()

    #cur.execute("DROP TABLE IF EXISTS conversations")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(50),
            user_id VARCHAR(50),
            user_message TEXT,
            bot_response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id SERIAL PRIMARY KEY,
            start_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_timestamp TIMESTAMP
        )
    """)

    conn.commit()
    cur.close()
    conn.close()

chatbot = Flask(__name__)
chatbot.secret_key = 'michaelramsay_secret2'
CORS(chatbot)
init_db()  # Initialize the database

def generate_unique_id():
    return str(uuid.uuid4())

# Add this part for affiliate keywords

affiliate_keywords = {
    "Booking.com": "You can book here with <a href='https://www.booking.com/'>Booking.com</a>",
    "Airbnb": "Check out options on <a href='https://www.airbnb.com/'>Airbnb</a>",
    "Expedia": "Find deals on <a href='https://www.expedia.com/'>Expedia</a>",
    "TripAdvisor": "Read reviews on <a href='https://www.tripadvisor.com/'>TripAdvisor</a>",
    "Kayak": "Compare prices on <a href='https://www.kayak.com/'>Kayak</a>",
    "Skyscanner": "Search for flights on <a href='https://www.skyscanner.net/'>Skyscanner</a>",
    "Hotels.com": "Find hotels at <a href='https://www.hotels.com/'>Hotels.com</a>",
    "Trivago": "Compare hotel prices on <a href='https://www.trivago.com/'>Trivago</a>",
    "Orbitz": "Find various travel deals on <a href='https://www.orbitz.com/'>Orbitz</a>",
    "Priceline": "Get discounted rates on <a href='https://www.priceline.com/'>Priceline</a>"
}

# Predefined answers in a dictionary

predefined_answers = {
    
    "Fuck ": "Inappropriate content detected.",
    "Shit": "Inappropriate content detected."
       }

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(predefined_answers.keys())

#chatbot = Flask(__name__)
#chatbot.secret_key = 'your_secret_key_here'  # Replace with your actual secret key

@chatbot.route('/', methods=['GET'])
def home():
    return render_template('frontpage.html')

@chatbot.route('/image/<path:filename>')
def serve_image(filename):
    return send_from_directory('image', filename)

@chatbot.before_request
def setup_conversation():
    # Check for the unique ID cookie to identify returning users
    user_id = request.cookies.get('user_id')
    if not user_id:
        print("No user ID cookie found, setting new one")
        user_id = generate_unique_id()
        session['returning_user'] = False
        response = make_response(render_template('frontpage.html'))
        response.set_cookie('user_id', user_id, max_age=60*60*24*365*2)  # Expires in 2 years
        return response
    else:
        print("Existing user with ID:", user_id)
        # If there's an existing session, just continue with it
        if 'conversation' not in session or session.get('cleared', False):
            print("New session being initialised for existing user")
            session['conversation'] = []
            session['awaiting_decision'] = False
            session['conversation_status'] = 'new'
            session['cleared'] = False
            session['returning_user'] = True
        else:
            print("Continuing existing session for user")
            session['returning_user'] = True

limiter = Limiter(
    app=chatbot, 
    key_func=get_remote_address
)

@limiter.request_filter
def exempt_users():
    return False  # return True to exempt a user from the rate limit

@limiter.limit("20 per minute; 50 per 10 minutes; 100 per hour")
def custom_limit_request_error():
    return jsonify({"answer": "Too many requests, please try again later"}), 429

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
    user_id = request.cookies.get('user_id')
    threshold = 0.9
    query = request.json.get('query')
    max_tokens = 50
    tokens = query.split()
    exit_words = ["exit", "quit", "bye", "goodbye"] ##why is this repeated? which set is being used?
    session['conversation'].append({"role": "user", "content": query})

    if any(word.lower() in query.lower() for word in exit_words):
        goodbye_message = "Thank you for your visit. Have a wonderful day. Goodbye!"
        # Reset the session keys instead of clearing everything.
        session['returning_user'] = False
        session['awaiting_decision'] = False
        session['conversation_status'] = 'new'
        session['cleared'] = True
        session['conversation'] = []
        
        return jsonify({"answer": goodbye_message, "status": "end_session"})
            
    if len(tokens) > max_tokens:
        answer = "Your query is too long. Please limit it to 50 words or less."
        return jsonify({"answer": answer})

    transcribed_text = session.get('transcribed_text', None)
    if transcribed_text:
        query = transcribed_text
        #del session['transcribed_text']

    query_vector = vectorizer.transform([query])
    
    if session.get('returning_user', False) and session.get('awaiting_decision', True):
        session['conversation_status'] = 'active'
        session['conversation'] = [
            {"role": "system", "content": "You are an advanced AI consultant at TalkAI Global, specialising in providing expert advice on AI-driven business solutions. Your key responsibilities include understanding client needs, explaining the benefits of AI technologies like chatbots and process automation, and guiding businesses towards effective AI integration. You aim to be concise and clear in your conversation. You never overwhelm the client with too much information in a single response. "}
        ]
        return_message = "Alright, let's start a new conversation."
    
   
    #else session.get('conversation_status', 'active') == 'active':
    else:
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
        
        conversation_with_prompt = [custom_prompt] + session['conversation']

        api_endpoint = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4-1106-preview",
            "messages": conversation_with_prompt, ######
            "frequency_penalty": 1.0,
            "presence_penalty": -0.5
        }
        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content']
            forbidden_phrases = ["I am a model trained", "As an AI model", "My training data includes", "ChatGPT","OpenAI"]
            for phrase in forbidden_phrases:
                answer = answer.replace(phrase, "")
        else:
            answer = "I'm sorry, I couldn't understand the question."

        session['conversation'].append({"role": "assistant", "content": answer})
        session.modified = True #
        print("After appending assistant answer:", session['conversation'])
 # Database interaction to save the conversation
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'], sslmode='require')
        cur = conn.cursor()
        
        # Insert the conversation into the database
        # Assuming session_id is being tracked, replace with actual session_id or NULL
        cur.execute(
            "INSERT INTO conversations (user_id, session_id, user_message, bot_response) VALUES (%s, %s, %s, %s)",
            (user_id, session.get('session_id'), query, answer)  # Replace with actual session logic
        )
        conn.commit()
        
    except Exception as e:
        print(f"Database error: {e}")
    finally:
        cur.close()
        conn.close()

    # Return the bot response
    return jsonify({"answer": answer})         

from datetime import timedelta

# set session timeout
chatbot.permanent_session_lifetime = timedelta(minutes=5)

            
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    chatbot.run(host='0.0.0.0', port=port)
