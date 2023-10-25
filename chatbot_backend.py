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


chatbot = Flask(__name__)
chatbot.secret_key = 'michaelramsay_secret' #for sessions

from flask_cors import CORS # for CORS

CORS(chatbot)



def custom_limit_request_error():
    return jsonify({"message": "Too many requests, please try again later"}), 429

limiter = Limiter(
    chatbot, 
    key_func=get_remote_address, 
    request_limit_exceeded=custom_limit_request_error
)

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
    
    "What is the currency in Jamaica?": "The currency in Jamaica is the Jamaican Dollar.",
    "What is the capital of Jamaica?": "The capital of Jamaica is Kingston."
    }

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(predefined_answers.keys())

#chatbot = Flask(__name__)
#chatbot.secret_key = 'your_secret_key_here'  # Replace with your actual secret key

@chatbot.route('/', methods=['GET'])
def home():
    return render_template('chatbot1.html')

@chatbot.route('/image/<path:filename>')
def serve_image(filename):
    return send_from_directory('image', filename)

@chatbot.before_request
def setup_conversation():
    if 'conversation' not in session:
        print("New session being initialized")
        session['conversation'] = [
            {"role": "system", "content": "You are a helpful assistant named Michael focused on Jamaica. You are a Rasta Jamaican. Your role is to assist the user with accurate and informative responses."}
        ]
    else:
        print("Existing session found")
    print("Initial session:", session.get('conversation'))

@limiter.limit("5 per minute")

@chatbot.route('/ask', methods=['POST'])
def ask():
    system_message = {}
    threshold = 0.7
    query = request.json.get('query')
    print("User query:", query)

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
        custom_prompt = {"role": "system", "content": "You are a helpful assistant with expertise on Jamaica. Your primary role is to assist the user by providing accurate and informative responses. It's essential that you maintain the context of the ongoing conversation, incorporating previous questions and answers to create a coherent and seamless dialogue. Each of your responses should logically follow from or relate to what has been previously discussed. This will ensure that the conversation flows naturally and that the user receives the most contextually relevant and helpful information."}
        # Add custom prompt to the beginning of the conversation history
        conversation_with_prompt = [custom_prompt] + session['conversation']
      
        # Use the conversation history for context-aware API call
        payload = {
            "model": "gpt-4",
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
    return jsonify({"answer": answer})

from datetime import timedelta

# set session timeout
chatbot.permanent_session_lifetime = timedelta(minutes=5)

    
            
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    chatbot.run(host='0.0.0.0', port=port)
