# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 20:27:48 2021

@author: Samuel Gandy
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



def countClass(data):
    print("CountData")
    
    classes = ['1','2','3','4','5','6','7'] #Set class numbers in array
    count = 0
    array = []
    for c in classes:                       #
        for i in range(0,len(data),1):
            #print(data[i], c)
            if int(data[i]) == int(c):
                count = count + 1
        
        #print(c, " = ", count)
        array.append(count)
        count = 0
    
    print(array)
    
    r = []
    
    for i in array:
        r.append((i/1357)*100)
                
    print(r)


def weighUpPred(results, proba):
    print("Weighing Up Results")
    print(results)
    print(proba)
    
    l = len(results)
    weight = float(l/100)
    print(weight)
    
    arr = []
    amount = 0
    
    for i in range(0,len(results),1):
        amount = exp(results[i])
    
    for r in results:
        arr.append(exp(r)/amount)
        
    #https://machinelearningmastery.com/softmax-activation-function-with-python/#:~:text=Softmax%20is%20a%20mathematical%20function,each%20value%20in%20the%20vector.&text=Each%20value%20in%20the%20output,of%20membership%20for%20each%20class.    
    
    print("result", arr)
    #Unsure how this will work 
    
  
#help from https://www.programiz.com/python-programming/writing-csv-files 
def CSVResults(predictions, name):                                #Function to plot results into a .csv file
    j = 0                                                   #Set the inital j value to 0 to start to the top of column
    with open(name,'w',newline='') as file:                 #Open the results file as a write
        w = csv.writer(file)                                #Set the cursor
        w.writerow(['indexOfTestSample','TypeOfDefects'])   #Find the rows
        for i in predictions:                               #run trough all the predictions                      
            j = j + 1                                       #Add 1 to j
            w.writerow([j, i])                              #Write the prediction at the row j
            
    print(len(predictions))                                 #Print predictions length
    
def reviewFeatures():
    pass
    

trainDF = pd.read_csv('Train.csv')          #Read the Traindata and import into dataFrame
testDF = pd.read_csv('Test.csv')            #Read the Testing data and import into dataFrame

trainDF = pd.get_dummies(trainDF)           #Normalise the train Data
testDF = pd.get_dummies(testDF)             #Normalise the test Data

print(trainDF)
print(testDF)


trainLabels = trainDF['TypeOfDefects']                  #Setting the labels to the labels column
trainFeatures = trainDF.drop(columns=['TypeOfDefects']) #Everything but the labels is the features

trainL = trainLabels    #Store the labels before the spilt
trainF = trainFeatures  #Store the features before the spilt

print(trainFeatures)
print(trainLabels)

#Feature Extraction

#trainFeatures = trainDF[['Pixels_Areas','X_Perimeter','Y_Perimeter','Sum_of_Luminosity','LogOfAreas','Log_X_Index','Log_Y_Index']]

#testDF = testDF[['Pixels_Areas','X_Perimeter','Y_Perimeter','Sum_of_Luminosity','LogOfAreas','Log_X_Index','Log_Y_Index']]

#Data is imported correctly now change type for NN

trainFeatures = np.array(trainFeatures) #Convert the trainFeatures into a numpy array
trainLabels = np.array(trainLabels)     #Convert the labels into a numpy array


print(trainFeatures[0:5])
print(trainLabels[0:5])

countClass(trainLabels)                     #Count the classes

#This is to make the softmax work
for i in range(0, len(trainLabels),1):      #Loop through every element
    trainLabels[i] = trainLabels[i] - 1     #Takeway one from each label

print(trainLabels)

print(trainFeatures.shape)
print(trainLabels.shape)

trainFeatures, valFeatures, trainLabels, valLabels = train_test_split(trainFeatures, trainLabels, test_size=0.1) #Spilting the data into train and validation

#Now the mechine learning

print("Mechine learning part")

depthOpt = [ 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50]                                     #Set the optimisation list for the decision tree
leadOpt = [ 5, 10, 15, 20, 25, 30, 40, 50, 100, 125, 130, 140, 130, 150, 175, 200, 250]     #Set the optimisation list for the decision tree
ranOpt = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 100, 250, ]                            #Set the optimisation list for the decision tree

minSpiltOpt = [2,4,6,8,10]              #Set the optimisation list for the decision tree
criterianOpt = ['gini','entropy']       #Array with the two types of decision tree


trainMax = 0
valMax = 0

resultTrain = []   #Declare the train result and result validation to be compared to find the right parameters
resultVal = []     

for i in depthOpt:
    for j in leadOpt:
        for g in ranOpt:
            for m in minSpiltOpt:
                for c in criterianOpt:
                    model = DecisionTreeClassifier(max_depth=i,max_leaf_nodes=j, random_state=g, min_samples_split=m, criterion=c) #Create Decision tree with the current parameters
                    model.fit(trainFeatures, trainLabels)               #Uses the current decision tree and train the model
                    trainScore = model.score(trainFeatures,trainLabels) #Find the train score
                    valScore = model.score(valFeatures, valLabels)      #find the validation score
                    
                    resultTrain.append(trainScore) #Add the result in array
                    resultVal.append(valScore)     #Add the validation result to array for graphing
                    
                    if trainScore > trainMax:       #If the train score is the highest 
                        trainMax = trainScore       #Set the current trainScore to the max
                        print("New train Record ", i, j, g, m, c, "Train Score is ", trainScore) #Print the new record
                        pickle.dump(model, open("BestTrainDT.sav", 'wb'))                        #Save the model
                    
                    if valScore > valMax:          #If the validation data is the highest
                        valMax = valScore          #set the current score to the maxium value
                        print("New Validation Record ", i, j, g, m, c, "Validation is ", valScore) #print the validation score
                        pickle.dump(model, open("BestValidationDT.sav", 'wb'))     #save the model
                        D = i #Save the parameters for the final model to be created
                        L = j
                        R = g
                        C = c
                        M = m
  

plt.scatter(resultTrain,resultVal)
plt.show()



#newModel = pickle.load(open("BestValidationDT.sav",'rb'))


newModel =  DecisionTreeClassifier(max_depth=D,max_leaf_nodes=L, random_state=R, min_samples_split=M, criterion=C) #Create a new model with the best parameters
newModel.fit(trainFeatures, trainLabels)                    #Train the new model

trainedScore = newModel.score(trainFeatures,trainLabels)    #get the score for the trainning data
print(trainedScore)                                         #print the results of the trainning data

valScore = newModel.score(valFeatures, valLabels)           #find the score for the validation data
print(valScore)                                             #print the validation score

pred = newModel.predict(testDF)                             #Use the new model to predict the testing data
print(pred)                                                 #print predictions

results = []

for i in range(0, len(pred),1):  #Run through the predictions array
    r = int(pred[i]) + int(1)    #Plus 1 to the array as the final score is starts at 1 and not 0
    results.append(r)            #add to the end of the array
    
   
print(results)                              #print the results
CSVResults(results, "InitalResults.csv")    #save the predictions in CSV file

print("Vsiaulise the Tree")

text_representation = tree.export_text(newModel) #show text version of final decsion tree

print(text_representation)  #print the decision tree

print("Save the Model")

pickle.dump(newModel, open("DefectsModel.sav", 'wb')) #Save the model


#load the model

newModel = pickle.load(open("DefectsModel.sav",'rb')) #Load up the saved model
result = newModel.score(valFeatures, valLabels)         #Get the score of the model
print(result)                                           #print the results