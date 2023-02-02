# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 14:01:08 2021

@author: Samuel Gandy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

print("PCA")

trainDF = pd.read_csv('Train.csv')
testDF = pd.read_csv('Test.csv')

print(trainDF)
print(testDF)

trainLabels = trainDF['TypeOfDefects']
trainFeatures = trainDF.drop(columns=['TypeOfDefects'])

print(trainFeatures)
print(trainLabels)

trainFeatures = trainDF[['Pixels_Areas','X_Perimeter','Y_Perimeter','Sum_of_Luminosity','LogOfAreas','Log_X_Index','Log_Y_Index']]

testDF = testDF[['Pixels_Areas','X_Perimeter','Y_Perimeter','Sum_of_Luminosity','LogOfAreas','Log_X_Index','Log_Y_Index']]

#I followed this tutorial https://github.com/krishnaik06/Principle-Component-Analysis 
scaler = StandardScaler()
scaler.fit(trainDF)

scaled_data = scaler.transform(trainDF)

print(scaled_data)

pca = PCA(n_components = 2)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=trainLabels)
plt.xlabel('First principle component')
plt.ylabel('Second principle component')