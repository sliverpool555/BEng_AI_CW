# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:39:42 2021

@author: Samuel Gandy


File Extraction Program 
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


def commparePlot(arrayA, arrayB):       #Declare Function to compare variables
    print("Comparing two variables")
    plt.scatter(arrayA,arrayB)          #Plot the variables against each other
    plt.show()                          #show the graph
    

def CSVResults(predictions):                                #Function to plot results into a .csv file
    j = 0                                                   #Set the inital j value to 0 to start to the top of column
    with open("Results.csv",'w',newline='') as file:        #Open the results file as a write
        w = csv.writer(file)                                #Set the cursor
        w.writerow(['indexOfTestSample','TypeOfDefects'])   #Find the rows
        for i in predictions:                               #run trough all the predictions                      
            j = j + 1                                       #Add 1 to j
            w.writerow([j, i])                              #Write the prediction at the row j
            
    print(len(predictions))                                 #Print predictions length
    


trainDF = pd.read_csv('Train.csv')                          #Read the Traindata and import into dataFrame
testDF = pd.read_csv('Test.csv')                            #Read the Testing data and import into dataFrame

print(trainDF)
print(testDF)

trainLabels = trainDF['TypeOfDefects']                      #read class column and set to the labels
trainFeatures = trainDF.drop(columns=['TypeOfDefects'])     #Features are the columns excluding the classes



print(trainFeatures)
print(trainLabels)

print(trainFeatures.describe(include='all'))    #Show the dataset is been read

#Feature Extraction of all features

print("Feature Extraction")

features = testDF.keys()            #Read the features

print(features)
print(len(features))

for i in features:  #print the features indevualy
    print(i)


X_Minimum = trainDF['X_Minimum']    #Read each inderval column to get only that data from that feature

X_Maximum = trainDF['X_Maximum']

Y_Minimum = trainDF['Y_Minimum']

Y_Maximum = trainDF['Y_Maximum']

Pixels_Areas = trainDF['Pixels_Areas']

X_Perimeter = trainDF['X_Perimeter']

Y_Perimeter = trainDF['Y_Perimeter']

Sum_of_Luminosity = trainDF['Sum_of_Luminosity']

Minimum_of_Luminosity = trainDF['Minimum_of_Luminosity']

Maximum_of_Luminosity = trainDF['Maximum_of_Luminosity']

Length_of_Conveyer = trainDF['Length_of_Conveyer']

TypeOfSteel_A300 = trainDF['TypeOfSteel_A300']

TypeOfSteel_A400 = trainDF['TypeOfSteel_A400']

Steel_Plate_Thickness = trainDF['Steel_Plate_Thickness']

Edges_Index = trainDF['Edges_Index']

Empty_Index = trainDF['Empty_Index']

Square_Index = trainDF['Square_Index']

Outside_X_Index = trainDF['Outside_X_Index']

Edges_X_Index = trainDF['Edges_X_Index']

Edges_Y_Index = trainDF['Edges_Y_Index']

Outside_Global_Index = trainDF['Outside_Global_Index']

LogOfAreas = trainDF['LogOfAreas']

Log_X_Index = trainDF['Log_X_Index']

Log_Y_Index = trainDF['Log_Y_Index']

Orientation_Index = trainDF['Orientation_Index']

Luminosity_Index = trainDF['Luminosity_Index']

SigmoidOfAreas = trainDF['SigmoidOfAreas']



#commparePlot(trainLabels, SigmoidOfAreas)

plt.figure(figsize=(16,10))             #Set the size of the figure
sn.heatmap(trainDF.corr(),annot=True)   #get the correlations and print using sea born library
plt.show()                              #Show the data


print("Feature Selection") 

trainFeatures = trainDF[['Pixels_Areas','X_Perimeter','Y_Perimeter','Sum_of_Luminosity','LogOfAreas','Log_X_Index','Log_Y_Index']]  #set the features to the ones selected

testDF = testDF[['Pixels_Areas','X_Perimeter','Y_Perimeter','Sum_of_Luminosity','LogOfAreas','Log_X_Index','Log_Y_Index']]          #set the features to the ones selected

print(trainFeatures)

print(trainFeatures.shape)              #Print the shape

#This is to make the softmax work
for i in range(0, len(trainLabels),1):  #Run through each element in trainning labels
    trainLabels[i] = trainLabels[i] - 1 #Takeaway 1 from each label to make the classification work for decision trees
    
    
print(trainFeatures.shape)
print(trainLabels.shape)

trainFeatures, valFeatures, trainLabels, valLabels = train_test_split(trainFeatures, trainLabels, test_size=0.1) #Spilt the data into train and test
    
modelA = DecisionTreeClassifier(max_depth=5,max_leaf_nodes=10, random_state=5)  #Create the decision tree with set paramters
modelA.fit(trainFeatures, trainLabels)                                          #Train the Decision Tree
print("DT Trainning", modelA.score(trainFeatures,trainLabels))                  #print the trainning score

pred = modelA.predict(testDF)                                   #Get the predictions for the test dataframe


predictA = modelA.predict(trainFeatures)                        #Use decsion trees to Predict for the train features
matrix = confusion_matrix(trainLabels, predictA)                #Using prediction and correct labels to make a confussuin matrix
sn.heatmap(matrix, annot=True, cmap='Blues')                    #Show the confussion matrix

print("DT Validation", modelA.score(valFeatures, valLabels))    #Print the score of the validation data

results = []                        #declare the results array

for i in range(0, len(pred),1):     #Run through every element of the predictions
    r = int(pred[i]) + int(1)       #Add one to every class to bring it back to normal
    results.append(r)               #Add the plused one variable to the end of the array
    
print(results)
CSVResults(results)     #Using the CVSresults to print the results in csv



print("SVM Model")

clf = svm.SVC(gamma='scale')            #Build the svm model
clf.fit(trainFeatures, trainLabels)     #Train the svm model 

pred = clf.predict(testDF)              #Preditc using svm model
print(pred)

print("SVM Train", clf.score(trainFeatures,trainLabels))        #Print the trainning score

print("SVM Validation", clf.score(valFeatures, valLabels))      #Print the validation score



