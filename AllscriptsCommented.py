"""
@author: Samuel Gandy -up861111

Artificial Intelligence ENG621 
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
    
    classes = ['1','2','3','4','5','6','7']     #Set class numbers in array
    count = 0                                   #Set the intial value to 0
    array = []
    for c in classes:                           #loop through classes
        for i in range(0,len(data),1):          #Loop through the dataset
            #print(data[i], c)
            if int(data[i]) == int(c):          #If the data is equal to the class
                count = count + 1               #Add one to the count
        
        #print(c, " = ", count)
        array.append(count)                     #Add the count to array to store
        count = 0                               #Reset the count
    
    print(array)                                #Show array
    
    r = []                                      #Declare r array
    
    for i in array:                             #Loop through array
        r.append((i/1357)*100)                  #Add percentage to array
                
    print(r)                                    #print array of percentages

    
  
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

pickle.dump(newModel, open("DefectsModel.h5", 'wb')) #Save the model


#load the model

newModel = pickle.load(open("DefectsModel.h5",'rb')) #Load up the saved model
result = newModel.score(valFeatures, valLabels)         #Get the score of the model
print(result)



#Confussion Matrix Code

predictA = model.predict(trainFeatures)                        #Use decsion trees to Predict for the train features
matrix = confusion_matrix(trainLabels, predictA)                #Using prediction and correct labels to make a confussuin matrix
sn.heatmap(matrix, annot=True, cmap='Blues')                    #Show the confussion matrix



#Feature Correlation Code

plt.figure(figsize=(16,10))             	#Set the size of the figure
sn.heatmap(trainDF.corr(),annot=True)   	#get the correlations and print using sea born library
plt.show()                             		#Show the data


#Data Analysis Script

def countClass(data):
    print("CountData")
    
    classes = ['1','2','3','4','5','6','7']     #Set the classes within array
    count = 0                                   #inital count value to 0
    array = []
    for c in classes:                           #loop through variables in class
        for i in range(0,len(data),1):          #Loop through data indexs
            #print(data[i], c)
            if int(data[i]) == int(c):          #If the data equals the class set 
                count = count + 1               #Plus one
        
        #print(c, " = ", count)
        array.append(count)                     #add to array
        count = 0                               #reset the count to 0 for next class
    
    print(array)
    
    r = []                      #initalise r
    
    for i in array:             #loop through array
        r.append((i/1357)*100)  #Add percentage to array

    print(r)                    #print percentages



#PCA Script

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

print("PCA")

trainDF = pd.read_csv('Train.csv')  #Import trainning data into dataframe
testDF = pd.read_csv('Test.csv')    #import test data into dataframe

print(trainDF)
print(testDF)

trainLabels = trainDF['TypeOfDefects']                  #Find trainning label and store in array
trainFeatures = trainDF.drop(columns=['TypeOfDefects']) #everything but class is features

print(trainFeatures)
print(trainLabels)



scaler = StandardScaler()   #Define the scaler to standard scaler from Sklearn package
scaler.fit(trainDF)         #Fit the model

scaled_data = scaler.transform(trainDF) #add the trainning data to scale

print(scaled_data)

pca = PCA(n_components = 2) #Set PCA to two axis
pca.fit(scaled_data)        #Fit the scaled data

x_pca = pca.transform(scaled_data) #Get the data points within array


plt.figure(figsize=(8,6))                           #set the figure size
plt.scatter(x_pca[:,0],x_pca[:,1],c=trainLabels)    #plot the scatter data
plt.xlabel('First principle component')             #Label axis
plt.ylabel('Second principle component')

#Standard Vector Machine Script 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import csv
import pickle



def CSVResults(predictions):                                #Plot the results function
    j = 0
    with open("Results.csv",'w',newline='') as file:        #Open results csv and call the data file
        w = csv.writer(file)                                #Create cursor to write the data
        w.writerow(['indexOfTestSample','TypeOfDefects'])   #Declare the row names to be written
        for i in predictions:                               #Loop through predictions
            j = j + 1                                       #Increament j for the row number
            w.writerow([j, i])                              #Write the prediction in the element
            
    print(len(predictions))                                 #print amount of predictions


print("SVM MODEL")

trainDF = pd.read_csv('Train.csv')      #Reading data using panadas is commented above
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

clf = svm.SVC(gamma='auto')             #Intialise model with gamma configation to automatical optimise (aid from https://scikit-learn.org/stable/modules/svm.html)
clf.fit(trainFeatures, trainLabels)     #Plot data on the SVM model for the support vectors to learn

pred = clf.predict(testDF)              #Predict using the models
print(pred)

print("Features", clf.score(trainFeatures,trainLabels)) #Using features and labels find the accuracy of the model
    
CSVResults(pred)                                        #Using predictions to predict in the CSV fie

print("Validation", clf.score(valFeatures, valLabels))  #Find the validation score


#Save the model

pickle.dump(clf, open("DefectsModel.sav", 'wb'))        #Save the SVM model


#load the model

model = pickle.load(open("DefectsModel.sav",'rb'))      #Read the file
result = model.score(valFeatures, valLabels)            #use loaded model to score with validation
print(result)                                           #print results


#Plot Decision Tree

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
import graphviz


trainDF = pd.read_csv('Train.csv')      #Commented code for the extraction of data from csv above
testDF = pd.read_csv('Test.csv')

trainDF = pd.get_dummies(trainDF)
testDF = pd.get_dummies(testDF)

print(trainDF)
print(testDF)


trainLabels = trainDF['TypeOfDefects']
trainFeatures = trainDF.drop(columns=['TypeOfDefects'])

trainL = trainLabels
trainF = trainFeatures

print(trainFeatures)
print(trainLabels)

#This is to make the softmax work
for i in range(0, len(trainLabels),1):  #Loop through classes within labels
    trainLabels[i] = trainLabels[i] - 1 #Reduce the labels by one for decision tree format

print(trainLabels)

print(trainFeatures.shape) #print the shape of the features and labels
print(trainLabels.shape)

trainFeatures, valFeatures, trainLabels, valLabels = train_test_split(trainFeatures, trainLabels, test_size=0.1) #Spilt the train and validation

newModel = pickle.load(open("DefectsModel.h5",'rb'))   #Load the  model
result = newModel.score(valFeatures, valLabels)         #Use the score function to test model
print(result)                                           #Plot score

text_representation = tree.export_text(newModel)        #Export tress as text repreatention in console
print(text_representation)                              #Plot in console

fig = plt.figure(figsize=(20,20))                       #set figure size to fit tree on plot
tree.plot_tree(newModel, filled=True)                   #Plot the tree


