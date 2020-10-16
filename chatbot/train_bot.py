import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lematizer = WordNetLemmatizer()
import json
import pickle
import random
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

words = []
tags= []
documents = []
ignore_words = ['?','!']

data = open('intents.json').read()
intents = json.loads(data)
# print(intents)

                            #  Preprocess the DATA
# Tokenize words
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = word_tokenize(pattern)
        # Add to the end of the list words
        words.extend(w)
        #  add the patterns to corpus
        documents.append((w, intent['tag']))

        #  add the tags in a separate list
        if intent['tag'] not in tags:
# print(words)
# print(documents)
            tags.append(intent['tag'])
# print(tags)

# Lemmatizing the words for proper meaning and removing duplicates
# sort vocabulary and tags
words = sorted(list(set(words)))
tags = sorted(list(set(tags)))
words = [lematizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# add words and tags to a pickle file

# pickle.dump(words, open('words.pkl','wb'))
# pickle.dump(tags, open('tags.pkl','wb'))




      # create testing and trainig data in numbers because the machine can not read text

trainig= []
# Empty array for our labels
output = [0]* len(tags)

# bag of words
training = []
output_empty = [0]*len(tags)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lematizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[tags.index(doc[1])] = 1
    trainig.append([bag,output_row])
        

random.shuffle(trainig)
trainig = np.array(trainig)

train_x= list(trainig[:,0])
train_y = list(trainig[:,1])

        # CREATE our neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),) ,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# define optimizer and compile the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fit the model on training data
chatbot_model = model.fit(np.array(train_x), np.array(train_y), epochs= 200, batch_size= 5, verbose=1)
model.save('chatbot.h5', chatbot_model)

