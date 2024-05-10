# Description: This file is used to train the model using the intents.json file. The model is saved as
# chatbot_model.keras. The words and classes are saved as words.pkl and classes.pkl respectively. The model is
# trained using the training data and the accuracy is calculating. The model is saving in the chatbot_model.keras file.
# The accuracy is print on the console.

# -------------Importing the required libraries and modules-------------- #
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# ----------------------------------------------------------------------- #

# -------------Downloading the required files from NLTK------------------ #
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
# ----------------------------------------------------------------------- #

# ---------------------Initializing the required variables--------------- #
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# ----------------------------------------------------------------------- #

# ---------------------Reading the intents.json file--------------------- #
with open('Intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)
# ----------------------------------------------------------------------- #

# ---------------------Initializing the required arrays------------------ #
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']
# ----------------------------------------------------------------------- #

# ---------------------Extracting the required information--------------- #
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend([word for word in word_list if word not in stop_words])
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# ----------------------------------------------------------------------- #

# ---------------------Cleaning up the words and classes----------------- #
words = [lemmatizer.lemmatize(word).lower() for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))
# ----------------------------------------------------------------------- #

# ---------------------Saving the words and classes---------------------- #
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
# ----------------------------------------------------------------------- #

# ---------------------Initializing the training data-------------------- #
training = []
output_empty = [0] * len(classes)
# ----------------------------------------------------------------------- #

# ---------------------Creating the bag of words------------------------- #
for document in documents:
    bag = []
    pattern_words = document[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in words:
        bag.append(1 if word in pattern_words else 0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)
# ----------------------------------------------------------------------- #

# ---------------------Shuffling the training data----------------------- #
random.shuffle(training)
training = np.array(training)
# ----------------------------------------------------------------------- #

# ---------------------Splitting the training data----------------------- #
train_x = training[:, :len(words)]
train_y = training[:, len(words):]
# ----------------------------------------------------------------------- #

# ---------------------Creating the model architecture------------------- #
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(train_x[0]),)), 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])
# ----------------------------------------------------------------------- #

# ----------------------Compiling the model------------------------------ #
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# ----------------------------------------------------------------------- #

# -----------------------Initializing the callbacks---------------------- #
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
# ----------------------------------------------------------------------- #

# ----------------------Training the model------------------------------- #
hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1, callbacks=[early_stopping])
model.save('chatbot_model.keras')
# ----------------------------------------------------------------------- #

# ----------------------Calculating the accuracy------------------------- #
model_accuracy = model.evaluate(np.array(train_x), np.array(train_y))
# ----------------------------------------------------------------------- #

# ----------------------Printing the accuracy---------------------------- #
print('=======================================')
print('⚠️⚠️⚠️ Model Accuracy: {:.2f}%'.format(model_accuracy[1] * 100) + ' ⚠️⚠️⚠️')
print('⚠️⚠️⚠️ Model Loss: {:.2f}%'.format(model_accuracy[0] * 100) + ' ⚠️⚠️⚠️')
print('🚨🚨🚨🚨 Training completed! 🚨🚨🚨🚨')
print('=======================================')
# ----------------------------------------------------------------------- #