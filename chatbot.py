# Description: This file contains the code of access trained AI model and accessed through the flask API.

#-------------Importing the required libraries and modules--------------#
from flask import Flask, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from flask_cors import CORS
from flask_cors import cross_origin
import re
#-----------------------------------------------------------------------#

#----------------------Initializing the Flask app-----------------------#
app = Flask(__name__)
#-----------------------------------------------------------------------#

#---------------------------Enabling CORS-------------------------------#
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})
#-----------------------------------------------------------------------#

#----------------------Loading the required files-----------------------#
lemmatizer = WordNetLemmatizer()
intents = json.load(open('Intents.json', 'r', encoding='utf-8'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')
#-----------------------------------------------------------------------#

#----------------------Cleaning up the sentence-------------------------#
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())
    return [lemmatizer.lemmatize(word) for word in sentence_words]
#-----------------------------------------------------------------------#

#----------------------Creating the bag of words------------------------#
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)
#-----------------------------------------------------------------------#

#----------------------Predicting the class of the sentence-------------#
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
#-----------------------------------------------------------------------#

#--------Extracting the required information from the user input--------#
def extract_lodging_type(user_input):
    match = re.search(r'\b(hotel|hostel|resort)\b', user_input, re.IGNORECASE)
    return match.group(1).lower() if match else None

def extract_budget_amount(user_input):
    match = re.search(r'\b(\d+)\b', user_input)
    return match.group(1) if match else None

def extract_cuisine_type(user_input):
    match = re.search(r'\b(italian|chinese|indian|srilankan)\b', user_input, re.IGNORECASE)
    return match.group(1).lower() if match else None

def extract_activity_type(user_input):
    match = re.search(r'\b(beach|sightseeing|shopping|nightlife)\b', user_input, re.IGNORECASE)
    return match.group(1).lower() if match else None

def extract_accommodation_list(user_input):
    match = re.search(r'\b(cheap|luxury|expensive)\b', user_input, re.IGNORECASE)
    return match.group(1).lower() if match else None
#-----------------------------------------------------------------------#

#----------------Updating the context with the extracted information-----#
def update_context(user_input, context):
    lodging_type = extract_lodging_type(user_input)
    if lodging_type:
        context['lodging_type'] = lodging_type
    budget_amount = extract_budget_amount(user_input)
    if budget_amount:
        context['budget_amount'] = budget_amount
    cuisine_type = extract_cuisine_type(user_input)
    if cuisine_type:
        context['cuisine_type'] = cuisine_type
    activity_type = extract_activity_type(user_input)
    if activity_type:
        context['activity_type'] = activity_type
#-----------------------------------------------------------------------#

#----------------Filling the placeholders with the context values-------#
def fill_placeholders(response, context):
    placeholders = re.findall(r'\[(\w+)\]', response)
    for placeholder in placeholders:
        response = response.replace(f'[{placeholder}]', context.get(placeholder, 'N/A'))
    return response
#-----------------------------------------------------------------------#

#----------------Getting the response from the chatbot------------------#
def get_response(intents_list, intents_json, context):
    if not intents_list:
        return "I'm not sure how to respond to that. Could you rephrase it or ask something else?"
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            response = random.choice(i['responses'])
            return fill_placeholders(response, context)
    return "Sorry, I can't find an appropriate response."
#-----------------------------------------------------------------------#

#---------------------API endpoint for getting the response-------------#
@app.route('/get', methods=['POST'])
@cross_origin()
def get_bot_response():
    user_text = request.get_json().get('message')
    context = {}
    update_context(user_text, context)
    ints = predict_class(user_text)
    response = get_response(ints, intents, context)
    return jsonify({"response": response})
#-----------------------------------------------------------------------#

#----------------------Running the Flask app----------------------------#
if __name__ == "__main__":
    app.run(debug=True)
#-----------------------------------------------------------------------#
