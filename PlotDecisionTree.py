# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:51:11 2021

@author: Samuel Gandy


Plotting Decision Trees
"""


import numpy as np
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
import seaborn as sn
from math import exp
import csv
import pickle
from sklearn import svm




trainDF = pd.read_csv('Train.csv')              #Read the trainning data and import into dataFrame
testDF = pd.read_csv('Test.csv')                #Read the testing data and import into dataFrame

trainDF = pd.get_dummies(trainDF)               #Normalise the training data 
testDF = pd.get_dummies(testDF)                 #Normalise the testing data 

print(trainDF)
print(testDF)


trainLabels = trainDF['TypeOfDefects']                  #Setting the labels to the labels column
trainFeatures = trainDF.drop(columns=['TypeOfDefects']) #Everything else but label column is the features

trainL = trainLabels                        #Save the trainning data before the spilt
trainF = trainFeatures

print(trainFeatures)
print(trainLabels)

#This is to make the softmax work
for i in range(0, len(trainLabels),1):      #Run through all elements in labels array
    trainLabels[i] = trainLabels[i] - 1     #Takeway -1 from the class number so the labels are able to be trainned for the decision tree

print(trainLabels)

print(trainFeatures.shape)
print(trainLabels.shape)

trainFeatures, valFeatures, trainLabels, valLabels = train_test_split(trainFeatures, trainLabels, test_size=0.1) #Spilt the test and validate



featuresNames = trainFeatures.head()    #The name of features are the column titles
print()
features = []                           

for f in trainFeatures.columns:         #Run throug every column
    features.append(f)                  #Add the column to array
print(features)

newModel = pickle.load(open("DefectsModel.sav",'rb'))   #Load the saved model
result = newModel.score(valFeatures, valLabels)         #Score the model on the data to check model is working
print(result)

text_representation = tree.export_text(newModel)        #Access the text version of the model
print(text_representation)                              #print text version of teh decision tree

fig = plt.figure(figsize=(20,20))                       #Fix the size of the figure
tree.plot_tree(newModel, filled=True)                   #Plot the visual decision tree


