# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:58:16 2021

@author: Samuel Gandy
"""



import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
import seaborn as sn
from math import exp
import csv
import pickle
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV


def countClass(data):
    print("CountData")
    
    classes = ['1','2','3','4','5','6','7']
    count = 0
    array = []
    for c in classes:
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
def CSVResults(predictions):
    j = 0
    with open("Results.csv",'w',newline='') as file:
        w = csv.writer(file)
        w.writerow(['indexOfTestSample','TypeOfDefects'])
        for i in predictions:
            j = j + 1
            w.writerow([j, i])
            
    print(len(predictions))
    
    
def reviewFeatures():
    pass
    

trainDF = pd.read_csv('Train.csv')
testDF = pd.read_csv('Test.csv')

print(trainDF)
print(testDF)

trainDF = pd.get_dummies(trainDF)
testDF = pd.get_dummies(testDF)

trainLabels = trainDF['TypeOfDefects']
trainFeatures = trainDF.drop(columns=['TypeOfDefects'])

print(trainFeatures)
print(trainLabels)

#Feature Extraction

#trainFeatures = trainDF[['Pixels_Areas','X_Perimeter','Y_Perimeter','Sum_of_Luminosity','LogOfAreas','Log_X_Index','Log_Y_Index']]

#testDF = testDF[['Pixels_Areas','X_Perimeter','Y_Perimeter','Sum_of_Luminosity','LogOfAreas','Log_X_Index','Log_Y_Index']]

#Data is imported correctly now change type for NN

trainFeatures = np.array(trainFeatures)
trainLabels = np.array(trainLabels)


print(trainFeatures[0:5])
print(trainLabels[0:5])

countClass(trainLabels)

#This is to make the softmax work
for i in range(0, len(trainLabels),1):
    trainLabels[i] = trainLabels[i] - 1

print(trainLabels)

print(trainFeatures.shape)
print(trainLabels.shape)

trainFeatures, valFeatures, trainLabels, valLabels = train_test_split(trainFeatures, trainLabels, test_size=0.1)

#Now the mechine learning

print("Mechine learning part")

modelA = DecisionTreeClassifier(random_state=10)
modelA.fit(trainFeatures, trainLabels)
modelA.score(trainFeatures,trainLabels)

modelB = DecisionTreeClassifier(max_depth=5,max_leaf_nodes=20, random_state=10)
modelB.fit(trainFeatures, trainLabels)
modelB.score(trainFeatures,trainLabels)

modelC = DecisionTreeClassifier(max_depth=20,max_leaf_nodes=500, random_state=1)
modelC.fit(trainFeatures, trainLabels)
modelC.score(trainFeatures,trainLabels)

modelD = DecisionTreeClassifier(max_depth=15,max_leaf_nodes=100, random_state=1000)
modelD.fit(trainFeatures, trainLabels)
modelD.score(trainFeatures,trainLabels)


#Need a confussion matrix

print(modelA.score(trainFeatures, trainLabels))
print(modelB.score(trainFeatures, trainLabels))
print(modelC.score(trainFeatures, trainLabels))
print(modelD.score(trainFeatures, trainLabels))

#predictA = modelC.predict(valFeatures)

predA = modelA.predict(testDF)
predB = modelB.predict(testDF)
predC = modelC.predict(testDF)
predD = modelD.predict(testDF)

print(predA[:10])
print(predB[:10])
print(predC[:10])
print(predD[:10])

probaA = modelA.predict_proba(testDF)
probaB = modelB.predict_proba(testDF)
probaC = modelC.predict_proba(testDF)
probaD = modelD.predict_proba(testDF)

results = [predA[0], predB[0],predC[0],predD[0]]

probas = [probaA[0],probaB[0],probaC[0],probaD[0]]

result = weighUpPred(results,probas)


predictA = modelA.predict(valFeatures)
matrix = confusion_matrix(valLabels, predictA)
sn.heatmap(matrix, annot=True, cmap='Blues')


print(modelA.score(valFeatures, valLabels))
print(modelB.score(valFeatures, valLabels))
print(modelC.score(valFeatures, valLabels))
print(modelD.score(valFeatures, valLabels))

pred = modelB.predict(testDF)

results = []

for i in range(0, len(pred),1):
    r = int(pred[i]) + int(1)
    results.append(r)
    
print(results)
CSVResults(results)


#Save the model

pickle.dump(modelA, open("DefectsModel.sav", 'wb'))


#load the model

model = pickle.load(open("DefectsModel.sav",'rb'))
result = model.score(valFeatures, valLabels)
print(result)


""" Does not work 
print("Using Sklearn Opimization")

#print(DecisionTreeClassifier.get_params().keys())

parameters = {'max_depth' : (2,3,4,5,6,7,8,9,10,15,20,30,40,50,100),
              'max_leaf_nodes ' : (2,3,4,5,6,7,8,9,10,15,20,30,40,50,100),
              } #'criterion' : ('gini','entropy'), 'max_features ' : ('auto', 'sqrt', 'log2'), 'min_samples_split ' : (2,4,6)

DTmodel = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions = parameters)

DTmodel.fit(trainFeatures, trainLabels)

print(DTmodel.best_estimator)

#Build the model with the best parameters
"""

"""
#Improved model https://www.youtube.com/watch?v=HY2DcBhgwm0
model = DecisionTreeClassifier(random_state=10)
model.fit(trainFeatures, trainLabels)
model.score(trainFeatures,trainLabels)

#print(model.predict(testDF))

print(model.score(trainFeatures, trainLabels))

print(model.score(valFeatures, valLabels))

print(model.predict_proba(testDF))

"""


"""
new_model = DecisionTreeClassifier(max_depth=10,max_leaf_nodes=1000, random_state=10)
new_model.fit(trainFeatures, trainLabels)
new_model.score(trainFeatures,trainLabels)

#print(new_model.predict(testDF))

print(new_model.score(trainFeatures, trainLabels))

print(new_model.score(valFeatures, valLabels))

print(new_model.predict_proba(testDF))


tree.plot_tree(model)
#tree.plot_tree(new_model)
"""
