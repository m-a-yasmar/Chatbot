import os  # for environment variables
import json  # for JSON handling
import numpy as np  # for numerical operations
from sklearn.feature_extraction.text import TfidfVectorizer  # for TF-IDF
from sklearn.metrics.pairwise import cosine_similarity  # for cosine similarity
from flask import Flask, request, jsonify, render_template  # for Flask
import requests  # for HTTP requests
from flask import send_from_directory # To help insert image
from flask import session #for keeping history
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from uuid import uuid4
import psycopg2


from flask_cors import CORS # for CORS


def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = psycopg2.connect(os.environ['DATABASE_URL'], sslmode='require')
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            session_id INTEGER,
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
chatbot.secret_key = 'michaelramsay_secret'
CORS(chatbot)
init_db()  # Initialize the database



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
    if 'conversation' not in session:
        print("New session being initialized")
        session['conversation'] = [
            {"role": "system", "content": "You are an AI agent representing TalkAI Global, specializing in AI automation. Your primary role is to engage in a two-way conversation with users, focusing on understanding their needs and responding with insightful information about our AI services. Be concise yet informative, responding in a way that is not overwhelming. Ask relevant questions to gather user requirements and listen attentively to their queries. Provide brief, clear answers and encourage further questions or direct contact for detailed discussions, especially regarding pricing and service customization. Your aim is to create a connection by being an attentive listener and a knowledgeable guide in the world of AI solutions."}
        ]
    else:
        print("Existing session found")
    print("Initial session:", session.get('conversation'))
    
limiter = Limiter(
    app=chatbot,
    key_func=get_remote_address
)

@limiter.request_filter
def exempt_users():
    return False  # return True to exempt a user from the rate limit

#@limiter.limit("5 per minute")
@limiter.limit("5 per minute; 10 per 10 minutes; 20 per hour")
def custom_limit_request_error():
    return jsonify({"message": "Too many requests, please try again later"}), 429

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
    system_message = {}
    threshold = 0.7
    query = request.json.get('query')
    print("User query:", query)

    max_tokens = 20  # Set your desired limit
    tokens = query.split()
    if len(tokens) > max_tokens:
        answer = "Your query is too long. Please limit it to 20 words or less."
        return jsonify({"answer": answer})

    session['conversation'].append({"role": "user", "content": query})
    
    print("After appending user query:", session['conversation'])
    
    if len(query.split()) < 3:
        last_assistant_message = next((message['content'] for message in reversed(session['conversation']) if message['role'] == 'assistant'), None)
        print("Last assistant message:", last_assistant_message)
        
        if last_assistant_message:
            system_message = {
                "role": "system",
                "content": f"The user's query seems incomplete. Refer back to your last message: '{last_assistant_message}' to better interpret what they might be asking."
            }
            if system_message:
                session['conversation'].append(system_message)

            

    query_vector = vectorizer.transform([query])
    predefined_vectors = vectorizer.transform(predefined_answers.keys())
    similarity_scores = cosine_similarity(query_vector, predefined_vectors).flatten()
    max_index = similarity_scores.argmax()
    max_score = similarity_scores[max_index]

    if max_score >= threshold:
        most_similar_question = list(predefined_answers.keys())[max_index]
        answer = predefined_answers[most_similar_question]
    else:
        # If no predefined answer is found, call OpenAI API
        api_endpoint = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        custom_prompt = {"role": "system", "content":"You are an advanced AI agent for TalkAI Global, a pioneering company in AI automation. Your primary role is to assist users with comprehensive and accurate information about our AI services and products. Provide detailed explanations, tailored advice, and innovative solutions to inquiries related to autonomous systems, strategic AI consulting, and custom AI solutions. Your responses should be professional, informative, and reflect the cutting-edge nature of our services. Handle all queries with a focus on showcasing how TalkAI Global can empower businesses with AI technology. Support, educate, and inspire potential clients about the transformative potential of AI in their operations."} 
        # Add custom prompt to the beginning of the conversation history
        conversation_with_prompt = [custom_prompt] + session['conversation']
      
        # Use the conversation history for context-aware API call
        payload = {
            "model": "gpt-4-1106-preview",
            "messages": conversation_with_prompt,
            "frequency_penalty": 1.5,  
            "presence_penalty": -1
        }
        #             "max_tokens": 80
        # frequency -2 to 2. higher increase repetition of answer  presence -2 to 2. higher likely to switch topic
        #response = requests.post(api_endpoint, headers=headers, json=payload)
        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)  # 15-second timeout

        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content'].strip()
            # Remove any forbidden phrases
            forbidden_phrases = ["I am a model trained", "As an AI model", "My training data includes","As an artificial intelligence","ChatGPT","OpenAI"]
            for phrase in forbidden_phrases:
                answer = answer.replace(phrase, "")
        else:
            
            answer = "I'm sorry, I couldn't understand the question."
    for keyword, replacement in affiliate_keywords.items():
        if keyword.lower() in answer.lower():
            answer = answer.replace(keyword, replacement)
    session['conversation'].append({"role": "assistant", "content": answer})
    session.modified = True
    print("After appending assistant answer:", session['conversation'])
 # Database interaction to save the conversation
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'], sslmode='require')
        cur = conn.cursor()
        
        # Insert the conversation into the database
        # Assuming session_id is being tracked, replace with actual session_id or NULL
        cur.execute(
            "INSERT INTO conversations (session_id, user_message, bot_response) VALUES (%s, %s, %s)",
            (session.get('session_id'), query, answer)  # Replace with actual session logic
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
