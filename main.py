# Description: This file is used to train the model using the intents.json file. The model is saved as chatbot_model.keras. The words and classes are saved as words.pkl and classes.pkl respectively. The model is trained using the training data and the accuracy is calculated. The model is saved in the chatbot_model.keras file. The weights are saved in the Weights folder. The accuracy is printed on the console.

#-------------Importing the required libraries and modules--------------#
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
#-----------------------------------------------------------------------#

#-------------Downloading the required files from NLTK------------------#
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
#-----------------------------------------------------------------------#

#---------------------Initializing the required variables---------------#
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
#-----------------------------------------------------------------------#

#---------------------Reading the intents.json file---------------------#
with open('Intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)
#-----------------------------------------------------------------------#

#---------------------Initializing the required arrays------------------#
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']
#-----------------------------------------------------------------------#

#---------------------Extracting the required information---------------#
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend([word for word in word_list if word not in stop_words])
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
#-----------------------------------------------------------------------#

#---------------------Cleaning up the words and classes-----------------#
words = [lemmatizer.lemmatize(word).lower() for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))
#-----------------------------------------------------------------------#

#---------------------Saving the words and classes----------------------#
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
#-----------------------------------------------------------------------#

#---------------------Initializing the training data--------------------#
training = []
output_empty = [0] * len(classes)
#-----------------------------------------------------------------------#

#---------------------Creating the bag of words-------------------------#
for document in documents:
    bag = []
    pattern_words = document[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in words:
        bag.append(1 if word in pattern_words else 0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)
#-----------------------------------------------------------------------#

#---------------------Shuffling the training data-----------------------#
random.shuffle(training)
training = np.array(training)
#-----------------------------------------------------------------------#

#---------------------Splitting the training data-----------------------#
train_x = training[:, :len(words)]
train_y = training[:, len(words):]
#-----------------------------------------------------------------------#

#---------------------Creating the model architecture-------------------#
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])
#-----------------------------------------------------------------------#

#----------------------Compiling the model------------------------------#
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#-----------------------------------------------------------------------#

#----------------------Training the model-------------------------------#
hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)
model.save('chatbot_model.keras')
#-----------------------------------------------------------------------#

#----------------------Saving the weights-------------------------------#
checkpoint_name = 'Weights\Weights-{epoch:03d}--{val_accuracy:.5f}.keras'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = False, mode ='auto')
callbacks_list = [checkpoint]
#-----------------------------------------------------------------------#

#----------------------Calculating the accuracy-------------------------#
cmp_list=accuracy_score(np.argmax(train_y, axis=1), np.argmax(model.predict(train_x), axis=1))
#-----------------------------------------------------------------------#

#----------------------Printing the accuracy----------------------------#
print('Accuracy: ', cmp_list)
print('🚨🚨🚨🚨 Training completed! 🚨🚨🚨🚨')
#-----------------------------------------------------------------------#