# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 17:41:50 2021

@author: Student


"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import csv
import pickle



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


print("SVM MODEL")

trainDF = pd.read_csv('Train.csv')
testDF = pd.read_csv('Test.csv')

print(trainDF)
print(testDF)

trainLabels = trainDF['TypeOfDefects']
trainFeatures = trainDF.drop(columns=['TypeOfDefects'])

print(trainFeatures)
print(trainLabels)

trainFeatures, valFeatures, trainLabels, valLabels = train_test_split(trainFeatures, trainLabels, test_size=0.1)

clf = svm.SVC()
clf.fit(trainFeatures, trainLabels)

pred = clf.predict(testDF)
print(pred)

print("All features Predictions")
print(clf.score(trainFeatures,trainLabels))
    
CSVResults(pred)

print(clf.score(valFeatures, valLabels))


print()
print("Feature Extraction")

trainFeatures = trainFeatures[['Pixels_Areas','Sum_of_Luminosity','Minimum_of_Luminosity', 'Sum_of_Luminosity', 'Maximum_of_Luminosity','Steel_Plate_Thickness']]

testDF = testDF[['Pixels_Areas','Sum_of_Luminosity','Minimum_of_Luminosity', 'Sum_of_Luminosity', 'Maximum_of_Luminosity','Steel_Plate_Thickness']]

valFeatures = valFeatures[['Pixels_Areas','Sum_of_Luminosity','Minimum_of_Luminosity', 'Sum_of_Luminosity', 'Maximum_of_Luminosity','Steel_Plate_Thickness']]

print("SVC MODEL")

clf = svm.SVC(gamma='auto')
clf.fit(trainFeatures, trainLabels)

pred = clf.predict(testDF)
print(pred)

print("Features", clf.score(trainFeatures,trainLabels))
    
CSVResults(pred)

print("Validation", clf.score(valFeatures, valLabels))


#Save the model

pickle.dump(clf, open("DefectsModel.sav", 'wb'))


#load the model

model = pickle.load(open("DefectsModel.sav",'rb'))
result = model.score(valFeatures, valLabels)
print(result)




