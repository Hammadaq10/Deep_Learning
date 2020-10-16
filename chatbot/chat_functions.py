import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np 
import json
import random
from keras.models import load_model
model = load_model('chatbot.h5')

intents = json.loads(open('intents.json').read())
tags = pickle.load(open('tags.pkl','rb'))
words = pickle.load(open('words.pkl','rb'))

#  Sentence Cleaning Function

def clean_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    return sentence_words

#  Bag of Words funciton

def bagofwords(sentence, words, show_details=True):
    sentence_words = clean_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w==s:
                bag[i] = 1
                if show_details:
                    print("found in the bag: %s" % w)
    return (np.array(bag))

#  Prediction class
def prediction_class(sentence, model):
    bow = bagofwords(sentence,words, show_details=False)
    result = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(result) if r> ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for res in results:
        return_list.append({"intent": tags[res[0]], "probability": str(res[1])})
    return return_list

#  Get Response from the application
def getResponse(predicted_results, intents_json):
    tag = predicted_results[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

#  Response to the chatbot GUI
def respose_chatbot(message):
    predicted_results = prediction_class(message,model)
    response = getResponse(predicted_results,intents)

    return response



