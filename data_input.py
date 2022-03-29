import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_data(data):
    
    X=[]
    for i in data['quizzes']:
        quiz=np.array([int(j) for j in i]) #read quiz string as an array of integers
        X.append(quiz) 

    X=np.array(X)   
    X=X/9 #normalize X 

    y=[]
    for i in data.solutions:
        solution=np.array([int(j) for j in i])-1
        #read solution string as an array of integers
        # if input are numbers they have to start with 0 
        # otherwise in this case to_categorical creates for each entry 10 different possible outcomes of a cell 
        # and output shape of the prediction does not fit with this shape
        y.append(solution)
    
    y=np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    #make a train-test-split

    return X_train, X_test, y_train, y_test

def get_data2(data):
    
    X=[]
    for i in data['quizzes']:
        quiz=np.array([int(j) for j in i]) #read quiz string as an array of integers
        X.append(quiz) 

    X=np.array(X)   
    X=X/9 #normalize X 

    y=[]
    for i in data.solutions:
        solution=np.array([int(j) for j in i])-1
        #read solution string as an array of integers
        # if input are numbers they have to start with 0 
        # otherwise in this case to_categorical creates for each entry 10 different possible outcomes of a cell 
        # and output shape of the prediction does not fit with this shape
        y.append(solution)
    
    y=np.array(y)

    return X, y

