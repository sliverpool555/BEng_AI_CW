# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 14:51:43 2021

@author: Samuel Gandy
"""



import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers import Embedding
from sklearn.model_selection import train_test_split


trainDF = pd.read_csv('Train.csv')
testDF = pd.read_csv('Test.csv')

print(trainDF)
print(testDF)

trainLabels = trainDF['TypeOfDefects']
trainFeatures = trainDF.drop(columns=['TypeOfDefects'])

print(trainFeatures)
print(trainLabels)

#Data is imported correctly now change type for NN

trainFeatures = np.array(trainFeatures)
trainLabels = np.array(trainLabels)


print(trainFeatures[0:5])
print(trainLabels[0:5])

print(trainLabels)

print(trainFeatures.shape)
print(trainLabels.shape)


print("Scale the data")

trainFeatures, valFeatures, trainLabels, valLabels = train_test_split(trainFeatures, trainLabels, test_size=0.1)


# Classification neural network with Keras
model = Sequential()
model.add(Dense(8, input_shape = (27,)))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(7, activation = 'softmax'))





"""
EMBEDDING_DIM = 27

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=))
model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())


#Now trainning the model

history = model.fit(trainFeatures, trainLabels, batch_size=128, epochs=10,verbose=2,validation_split=0.2)

score, acc = model.evaluate(valFeatures, valLabels, batch_size=128, verbose=2)

print("Accuracy is ", acc)
"""