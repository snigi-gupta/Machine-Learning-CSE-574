#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

path_to_file = "wdbc.dataset"

# extracting data
data = pd.read_csv(path_to_file, encoding='UTF-8',header=None)
# print(type(data))


# In[20]:


# function to process the data
def data_preprocessor(data=None):
    # dropping column 1
    data = data.drop([0],axis=1)

    # get row, column count
    row_count = data.shape[0]
    col_count = data.shape[1]
    print("No. of rows: ",row_count)
    print("No. of columns: ",col_count)
    print("------------------------------------------------------------------------")

    col_names = []
    # col_names = [col_names.append("col_"+str(i+1)) for i in range(col_count)]

    # create list of header names
    for i in range(col_count):
        col_names.append("col_"+str(i+1))
    # print(col_names)

    # update dataset with header names
    data.columns = col_names

    # Map label column to 0 and 1. 
    # 0: B (Benign), 1: M (Malingnant)
    data.col_1 = data.col_1.map({'M':1, 'B':0})

    # split data in train(80%),validate(10%) and test(10%) dataframes
    # get the X,Y where X is training set and Y is target vector.
    X_train, X_other, Y_train, Y_other = train_test_split(data.iloc[:,1:], data.col_1, train_size=0.8, random_state=50)
    X_validate, X_test, Y_validate, Y_test = train_test_split(X_other,Y_other,train_size=0.5,random_state=50)

    return X_train, Y_train, X_validate, Y_validate, X_test, Y_test, data

# function to print shape
def print_shape(matrix=None,name=None):
    if type(matrix) is list:
        for m in range(len(matrix)):
            print("Shape of {}:        \t".format(name[m]),matrix[m].shape)
    else:
        print("Shape of {}:        \t".format(name),matrix.shape)
    
    print("------------------------------------------------------------------------")
        

# function to normalize the data
def normalize(X_train=None, Y_train=None, X_validate=None, Y_validate=None, X_test=None, Y_test=None):
    # normalize the feature set
    scaler_obj = StandardScaler()
    scaler_obj.fit(X_train)
    mean_val = scaler_obj.mean_
    # print(mean_val)
    X_train = scaler_obj.transform(X_train)
    X_validate = scaler_obj.transform(X_validate)
    X_test = scaler_obj.transform(X_test)

    # Converting Y into array values (did this to fix Y datas and make them into a vector)
    Y_train = np.array(Y_train)
    Y_validate = np.array(Y_validate)
    Y_test = np.array(Y_test)
        
    # reshaping Y target vectors from (x,) to (x,1)
    Y_train = Y_train.reshape(455,1)
    Y_validate = Y_validate.reshape(57,1)
    Y_test = Y_test.reshape(57,1)
    
    return X_train, Y_train, X_validate, Y_validate, X_test, Y_test

# function to initialize parameters. 
def initializer(hp_flag=False):
    # initialize weight with 0
    weight = np.zeros((30,1), dtype=np.int64)
#     print(weight)
#     print(len(weight))

    # initialize bias with 0
    bias = 0

    if hp_flag:
        # set learning rate to 
        learning_rate = [0.1, 0.01, 0.003, 0.005, 0.0001]
    else:
        # set learning rate to 
        learning_rate = 0.003
    
    print("Learning rate(s)", learning_rate)

    return weight, bias, learning_rate

# function to calculate sigmoid
def sigmoid_function(z):
    return 1/(1+np.exp(-z))

# function to calculate cross entropy
def cross_entropy_function(y,a):
    numerator = np.sum(y*np.log(a) + (1-y)*np.log(1-a))
    denominator = -y.shape[1]
    loss = numerator/denominator
    return loss

# function to calculate dw weight
def calculate_dw(X,Y,a):
    # print("Before calculating derivative of loss function w.r.t weight")
    # print_shape(matrix=[X,Y,a], name=['X_train','Y','a'])
    dw = np.dot(Y-a,X)/-Y.shape[1]
    # print_shape(matrix=dw,name='dw')
    return dw

# function to calculate db bias
def calculate_db(Y,a):
    # print("Before calculating derivative of loss function w.r.t bias")
    # print_shape(matrix=[Y,a], name=['Y','a'])
    db = np.sum(Y-a)/-Y.shape[1]
    return db

# function to classify activation
def classify_activation(a):
    return [1 if i >= 0.5 else 0 for i in a[0]]


# In[1]:


"""
To produce the graph of Loss vs Epoch for different hyperparameters,
set hp_flag (hyperparameter flag) to 'True'.
Default value of hp_flag is False. Flag is initialized in the last cell of jupyter notebook.
"""

# function to perform logistic regression for different hyperparameters
def hyperparameter(X_train=None, Y_train=None, weight=None,bias=None,learning_rate=None):
    
    # loss tracker
    all_losses = []
    
    # dictionary to store data
    regression_values = {}
    
    for i, rates in enumerate(learning_rate):
        # counter
        count = 0
    
        losses = []
        
        # re-initializing weight and bias to 0, since iterating over new learning rate
        weight = np.zeros((30,1), dtype=np.int64)
        bias = 0
        
        print("------------------------------------------------------------------------")
        print("\nLearning Rate: ", rates)
        
        # beginning epoch loop
        for epoch in range(10000):

            """
            Performing forward pass for Training data.
            Here we calculate:
            X_train: The training set with 455 samples and 30 feature set
            Y_train: The target vector with 455 targets
            z: 
            activation: The activation or the prediction
            losses: The loss tracker for training data. This is a list.
            weight: 
            bias: 

            """
            # using the genesis function z=mx+c for training data
            z = np.dot(np.transpose(weight),np.transpose(X_train)) + bias

            # calculating activation/prediction for training data
            activation = sigmoid_function(z)

            # changing the shape of Y_train from (455,1) to (1,455) for ease of calculation
            Y_train = Y_train.reshape(1,455)

            # printing shape of z, activation and Y_train only once
            if count == 0:
                print_shape(matrix=[z,activation,Y_train],name=['z','activation','Y_train'])

            # classify activation in 0 and 1 for training data
            """
            if activation >= 0.5, classify 1
            if activation < 0.5, classify 0
            """  


            # calculate cross entropy and keep track of loss for training data
            losses.append(cross_entropy_function(Y_train,activation))

            # printing value of loss for every 1000 epochs
            if epoch%1000 == 0:
                print("\n\nTrain Loss Value[{}]:  \t".format(epoch), losses[count])

            # increasing counter value after performing forward pass for training
            count = count + 1

            """
            Performing backward pass for Training data.
            Here we calculate:
            weight: 
            bias: 

            """

            # update weights and bias for training data
            weight = np.transpose(weight) - rates * calculate_dw(X_train,Y_train,activation)
            bias = bias - rates * calculate_db(Y_train,activation)

            # reshaping weight since weight transpose is used during calculation of z
            weight = weight.reshape(30,1)
            #break
        all_losses.append(losses)
    
    plt.figure(figsize= (10,8))
    for x,val in enumerate(all_losses):
        plt.plot(np.arange(len(all_losses[x])), all_losses[x], label='Training Loss Track')
    plt.legend(learning_rate, loc='upper right')
    print("------------------------------------------------------------------------")
    print("Cross Entropy Loss vs Epochs")
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('Epochs')
    plt.show()
    
    # update dictionary with required values
    regression_values.update({'weight':weight, 'bias':bias})
    
    return regression_values


# In[2]:


# function to perform logistic regression
def logistic_regression(X_train=None, Y_train=None, X_validate=None, Y_validate=None, X_test=None, 
                        Y_test=None, weight=None,bias=None,learning_rate=None):
    # counter
    count = 0
    
    # loss tracker
    losses = []
    vlosses = []
    
    # decision boundary
    training_accuracy = []
    validation_accuracy = []
    
    # dictionary to store data
    regression_values = {}
    
    print("------------------------------------------------------------------------")
    print("\nLearning Rate: ", learning_rate)
    
    # beginning epoch loop
    for epoch in range(10000):
        
        """
        Performing forward pass for Training data.
        Here we calculate:
        X_train: The training set with 455 samples and 30 feature set
        Y_train: The target vector with 455 targets
        z: 
        activation: The activation or the prediction
        losses: The loss tracker for training data. This is a list.
        weight: 
        bias: 
        
        """
        # using the genesis function z=mx+c for training data
        z = np.dot(np.transpose(weight),np.transpose(X_train)) + bias
        
        # calculating activation/prediction for training data
        activation = sigmoid_function(z)
        
        # changing the shape of Y_train from (455,1) to (1,455) for ease of calculation
        Y_train = Y_train.reshape(1,455)

        # printing shape of z, activation and Y_train only once
        if count == 0:
            print_shape(matrix=[z,activation,Y_train],name=['z','activation','Y_train'])
            
        # classify activation in 0 and 1 for training data
        """
        if activation >= 0.5, classify 1
        if activation < 0.5, classify 0
        """  
        # calculating training data accuracy
        training_accuracy.append(accuracy_score(Y_train[0],classify_activation(activation)))

        
        # calculate cross entropy and keep track of loss for training data
        losses.append(cross_entropy_function(Y_train,activation))
        
        # printing value of loss for every 1000 epochs
        if epoch%1000 == 0:
            print("\n\nTrain Loss Value[{}]:  \t".format(epoch), losses[count])
        
        """
        Performing forward pass for Validation data.
        Here we calculate:
        X_validation: The validation set with 57 samples and 30 feature set
        Y_validation: The target vector with 57 targets
        z_valid: 
        activation_valid: The activation or the prediction
        vlosses: The loss tracker for validation data. This is a list.
        vweight: 
        vbias: 
        
        """
            
        # using the genesis function z=mx+c for validation data
        z_valid = np.dot(np.transpose(weight),np.transpose(X_validate)) + bias
        
        # calculating activation/prediction for validation data
        activation_valid = sigmoid_function(z_valid)
        
        # changing the shape of Y_validate from (455,1) to (1,455) for ease of calculation
        Y_validate = Y_validate.reshape(1,57)
        
        
        # printing shape of z_valid, activation_valid and Y_validate only once
        if count == 0:
            print_shape(matrix=[z_valid,activation_valid,Y_validate],name=['z_valid','activation_valid','Y_validate'])
        
        # classify activation in 0 and 1 for validation data
        """
        if activation >= 0.5, classify 1
        if activation < 0.5, classify 0
        """        
        # calculating validation data accuracy
        validation_accuracy.append(accuracy_score(Y_validate[0],classify_activation(activation_valid)))

        
        # calculate cross entropy and keep track of loss for validation data
        vlosses.append(cross_entropy_function(Y_validate,activation_valid))
        
        # printing value of loss for every 1000 epochs
        if epoch%1000 == 0:
            print("Validate Loss Value[{}]:  \t".format(epoch), vlosses[count])
        
        # increasing counter value after performing forward pass for both training and validation data
        count = count + 1
        
        """
        Performing backward pass for Training data.
        Here we calculate:
        weight: 
        bias: 
        
        """

        # update weights and bias for training data
        weight = np.transpose(weight) - learning_rate * calculate_dw(X_train,Y_train,activation)
        bias = bias - learning_rate * calculate_db(Y_train,activation)
        
        # reshaping weight since weight transpose is used during calculation of z
        weight = weight.reshape(30,1)
        #break
    
    # calculate precision and recall for training data
    training_precision = precision_score(Y_train[0],classify_activation(activation))
    training_recall = recall_score(Y_train[0],classify_activation(activation))

    
    # calculate precision and recall for validation data
    validation_precision = precision_score(Y_validate[0],classify_activation(activation_valid))
    validation_recall = recall_score(Y_validate[0],classify_activation(activation_valid))
   
    
    plt.figure(figsize= (10,8))
    plt.plot(losses, '-g', label='Training Loss Track')
    plt.plot(vlosses, '-b', label='Validation Loss Track')
    plt.legend(loc='upper right')
    print("------------------------------------------------------------------------")
    print("Cross Entropy Loss vs Epochs")
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('Epochs')
    plt.show()
    print("------------------------------------------------------------------------")
    print("Training accuracy vs Epochs")
    plt.figure(figsize= (10,8))
    plt.plot(training_accuracy, label='Training Accuracy')
    plt.plot(validation_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.show()
    
    # update dictionary with required values
    
    regression_values.update({'weight':weight, 'bias':bias, 'training_accuracy': training_accuracy, 
                              'training_precision': training_precision, 'training_recall': training_recall, 
                              'validation_accuracy': validation_accuracy, 'validation_precision': validation_precision, 
                              'validation_recall': validation_recall})
    
    return regression_values


# In[3]:


# function to test model based on testing data
def test_model(weight=None,bias=None,X_test=None,Y_test=None):
    
    # using the genesis function z=mx+c for testing data
    z_test = np.dot(np.transpose(weight),np.transpose(X_test)) + bias

    # calculating activation/prediction for testing data
    activation_test = sigmoid_function(z_test)
    
    # changing the shape of Y_test from (57,1) to (1,57) for ease of calculation
    Y_test = Y_test.reshape(1,57)
    
    # printing shape of z_test, activation_test and Y_test
    print_shape(matrix=[z_test,activation_test,Y_test],name=['z_test','activation_test','Y_test'])
    
    # calculating testing data accuracy, precision and recall
    testing_accuracy = accuracy_score(Y_test[0],classify_activation(activation_test))
    testing_precision = precision_score(Y_test[0],classify_activation(activation_test))
    testing_recall = recall_score(Y_test[0],classify_activation(activation_test))
    
    return testing_accuracy, testing_precision, testing_recall


# In[24]:


# set hyperparameter flag to True to view loss against hyperparameters, else default value is False.
hp_flag = False

# calling data_preprocessor function
X_train, Y_train, X_validate, Y_validate, X_test, Y_test, data = data_preprocessor(data=data)

# calling normalize function
X_train, Y_train, X_validate, Y_validate, X_test, Y_test, = normalize(X_train=X_train, Y_train=Y_train, X_validate=X_validate, 
                                                                      Y_validate=Y_validate, X_test=X_test, Y_test=Y_test)
print("Data is normalized!")
print("Printing Matrix dimensions of all sets")
print_shape(matrix=[X_train, Y_train, X_validate, Y_validate, X_test, Y_test],
            name=['X_train','Y_train','X_validate','Y_validate','X_test','Y_test'])

# calling initializer function
weight, bias, learning_rate = initializer(hp_flag=hp_flag)
print_shape(matrix=weight,name='weight')

if hp_flag:
    # calling hyperparameter function
    regression_values = hyperparameter(X_train=X_train, Y_train=Y_train, weight=weight, bias=bias, 
                                           learning_rate=learning_rate)
else:
    # calling logistic_regression function
    regression_values = logistic_regression(X_train=X_train, Y_train=Y_train, X_validate=X_validate,
                                        Y_validate=Y_validate, X_test=X_test, Y_test=Y_test, weight=weight,
                                        bias=bias,learning_rate=learning_rate)

    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print("Weight {0} \t Bias {1}".format(regression_values['weight'].shape,regression_values['bias']))
    print("Training Accuracy: {:0.2f}%".format(regression_values['training_accuracy'][-1]*100))
    print("Training Precision: {:0.2f}%".format(regression_values['training_precision']*100))
    print("Training Recall: {:0.2f}%".format(regression_values['training_recall']*100))
    print("Validation Accuracy: {:0.2f}%".format(regression_values['validation_accuracy'][-1]*100))
    print("Validation Precision: {:0.2f}%".format(regression_values['validation_precision']*100))
    print("Validation Recall: {:0.2f}%".format(regression_values['validation_recall']*100))

    # calling test model function
    accuracy, precision, recall = test_model(weight=regression_values['weight'],bias=regression_values['bias'],X_test=X_test,Y_test=Y_test)
    print("Test Accuracy: {:0.2f}%".format(accuracy*100))
    print("Test Precision: {:0.2f}%".format(precision*100))
    print("Test Recall: {:0.2f}%".format(recall*100))


# In[ ]:





# In[ ]:




