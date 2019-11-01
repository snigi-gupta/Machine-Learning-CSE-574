#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Read Fashion MNIST dataset
import os
import util_mnist_reader as mnist_reader
import pandas as pd
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop,Adadelta,Adam
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout


# function to print shape
def print_shape(matrix=None,name=None):
    if type(matrix) is list:
        for m in range(len(matrix)):
            print("Shape of {}:        \t".format(name[m]),matrix[m].shape)
    else:
        print("Shape of {}:        \t".format(name),matrix.shape)
    
    print("------------------------------------------------------------------------")

    
# weight initializer
def random_initializer(row=0,col=0):
    
    r = row
    c = col
    
    m = np.random.randn(r,c)
    return m


# function to calculate sigmoid
def sigmoid_function(z):
    return 1/(1+np.exp(-z))


def sigmoid_derivate_function(z):
    return z * (1 - z)

# function to calculate softmax
# do along y axis and keep the dimensions
def softmax_function(z):
    e = np.exp(z)# - np.max(z, axis=1, keepdims=True))
    return e/np.sum(e, axis=1, keepdims=True)


# function to normalise the data
def normalize(X_train=None,Y_train=None,X_test=None,Y_test=None):
    
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    
    return X_train/255.0, Y_train, X_test/255.0, Y_test


def forward_pass(dataset,weight,bias):
    
    # perform forward pass b/w i/p layer and hidden layer
    z = np.dot(dataset,weight) + bias
    activation = sigmoid_function(z)    
    return z, activation

def calculate_accuracy(X,Y,a2):
    true_positive = 0
    for i in range(Y.shape[0]):
        if Y[i].argmax() == a2[i].argmax():
            true_positive += 1
    # print(true_positive)
    accuracy = true_positive/len(X)
    # print(accuracy*100)
    return accuracy


# In[6]:


# get the data

X_train, Y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, Y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

# normalise the data, we divide the entire data set by 255(rgb)
X_train,Y_train,X_test,Y_test = normalize(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)
print_shape(matrix=[X_train,Y_train,X_test,Y_test],name=['X_train','Y_train','X_test','Y_test'])

# create validation set from train set, train set 55k and valid set 5k
X_train,Y_train = X_train[5000:],Y_train[5000:]
X_valid,Y_valid = X_train[:5000],Y_train[:5000]
print_shape(matrix=[X_train,Y_train,X_valid,Y_valid,X_test,Y_test],name=['X_train','Y_train','X_valid','Y_valid','X_test','Y_test'])


# In[7]:


# TASK 1

count = 0

# loss tracker
losses = []
vlosses = []

# decision boundary
training_accuracy = []
validation_accuracy = []

# # normalise the data, we divide the entire data set by 255(rgb)
# X_train,Y_train,X_test,Y_test = normalize(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)
# print_shape(matrix=[X_train,Y_train,X_test,Y_test],name=['X_train','Y_train','X_test','Y_test'])

# # create validation set from train set, train set 55k and valid set 5k
# X_train,Y_train = X_train[5000:],Y_train[5000:]
# X_valid,Y_valid = X_train[:5000],Y_train[:5000]

no_of_nodes = 128
learning_rate = 0.5

label_size = len(Y_train[0])

# initilize weights, bias and learnig rate
hi_layer_weights = random_initializer(row=len(X_train[0]),col=no_of_nodes)
hi_bias = random_initializer(row=1,col=no_of_nodes)

op_layer_weights = random_initializer(row=no_of_nodes,col=label_size)
op_bias = random_initializer(row=1,col=label_size)

print_shape(matrix=[hi_layer_weights,op_layer_weights,hi_bias,op_bias],name=['w1','w2','b1','b2'])

batch = 500

for epoch in range(25):
    
    x=0
    y=batch
    cost = 0
    cost_valid = 0
    accuracy_train_batch = 0
    indexes = np.random.permutation(len(X_train))
    batches = len(X_train)//batch
    for i in range(batches):
        
        X_train_batch = X_train[indexes[x:y]]
        Y_train_batch = Y_train[indexes[x:y]]
    
        # perform forward pass b/w i/p layer and hidden layer
        # z1 = np.dot(X_train_batch,hi_layer_weights) + hi_bias
        # a1 = sigmoid_function(z1)
        
        z1, a1 = forward_pass(X_train_batch,hi_layer_weights,hi_bias)
        #print_shape(matrix=[z1,a1],name=['z1','a1'])

        # perform forward pass b/w hidden layer and o/p layer
        # a1 is now X_train for the o/p layer
        # z2 = np.dot(a1,op_layer_weights) + op_bias
        # a2 = softmax_function(z2)

        z2, a2 = forward_pass(a1,op_layer_weights,op_bias)
        # print_shape(matrix=[z2,a2],name=['z2','a2'])
        
        """
        Cost function = -yloga
        derv_C/derv_op_layer_weights = weight change value
        derv_C/derv_op_bias = bias change value
        """

        log_of_a2 = np.log(a2)
        #print_shape(matrix=[log_of_a2,Y_train],name=["log_of_a2",'Y_train'])
        cost = cost -np.sum(Y_train_batch*log_of_a2)/len(X_train_batch)
        a2_dash = a2-Y_train_batch
        #print_shape(matrix=a2_dash,name='a2_dash')

        # perform backward pass b/w o/p layer and hidden layer
        dc_dw_op = np.dot(np.transpose(a1),a2_dash)
        dc_db_op = a2_dash # since dz2/db2 is 1
        #print_shape(matrix=[dc_dw_op,dc_db_op],name=['dc/dw2','dc/db2'])

        # update the weights
        #print("Shapes after updating weights\n")
        op_layer_weights = op_layer_weights - learning_rate * dc_dw_op/len(X_train_batch)
        op_bias = op_bias - learning_rate * np.sum(dc_db_op, keepdims=True, axis=0)/len(X_train_batch) # to convert into 1x10
        #print_shape(matrix=[op_layer_weights,op_bias],name=['w2','b2'])

        # perform backward pass b/w hidden layer and i/p layer
        dc_dw_hi = np.dot(np.transpose(np.dot(a2_dash,np.transpose(op_layer_weights)) * sigmoid_derivate_function(a1)),X_train_batch)
        dc_db_hi = np.dot(a2_dash,np.transpose(op_layer_weights)) * sigmoid_derivate_function(a1)
        #print_shape(matrix=[dc_dw_hi,dc_db_hi],name=['dc/dw1','dc/db1'])

        # update the weights
        #print("Shapes after updating weights\n")
        hi_layer_weights = hi_layer_weights - learning_rate * np.transpose(dc_dw_hi)/len(X_train_batch)
        hi_bias = hi_bias - learning_rate * np.sum(dc_db_hi)/len(X_train_batch)
        #print_shape(matrix=[hi_layer_weights,hi_bias],name=['w1','b1'])
        
        accuracy_train_batch = accuracy_train_batch + calculate_accuracy(X_train_batch,Y_train_batch,a2)
        x=x+batch
        y=y+batch
    losses.append(cost/batches)
    # accuracy of each batch
    training_accuracy.append(accuracy_train_batch/batches)
        
    if epoch%5 == 0:
        print("\n\nTrain Loss Value[{}]:  \t".format(epoch), losses[epoch])
        
    z1_valid =  np.dot(X_valid,hi_layer_weights) + hi_bias
    a1_valid = sigmoid_function(z1_valid)

    z2_valid = np.dot(a1_valid,op_layer_weights) + op_bias
    a2_valid = softmax_function(z2_valid)

    log_of_a2_valid = np.log(a2_valid)
    #print_shape(matrix=[log_of_a2,Y_train],name=["log_of_a2",'Y_train'])
    cost_valid = cost_valid -np.sum(Y_valid*log_of_a2_valid)/len(X_valid)
    vlosses.append(cost_valid)
    validation_accuracy.append(calculate_accuracy(X_valid,Y_valid,a2_valid))

print("Finished!")


# In[10]:


plt.figure()
plt.plot(losses, '-g', label='Training Loss Track')
plt.legend(loc='upper right')
print("------------------------------------------------------------------------")
plt.title("Loss vs Epochs")
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()

plt.figure()
plt.plot(vlosses, '-b', label='Validation Loss Track')
plt.legend(loc='upper right')
print("------------------------------------------------------------------------")
plt.title("Loss vs Epochs")
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()
# plt.plot(losses)
# plt.show()


# In[12]:


plt.figure()
plt.plot(training_accuracy, label='Training Accuracy')
plt.legend(loc='lower right')
print("------------------------------------------------------------------------")
plt.title("Accuracy vs Epochs")
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()



plt.figure()
plt.plot(validation_accuracy, 'c', label='Validation Accuracy')
plt.legend(loc='lower right')
print("------------------------------------------------------------------------")
plt.title("Accuracy vs Epochs")
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()


# In[13]:


z1_test =  np.dot(X_test,hi_layer_weights) + hi_bias
a1_test = sigmoid_function(z1_test)

z2_test = np.dot(a1_test,op_layer_weights) + op_bias
a2_test = softmax_function(z2_test)

true_positive = 0
for i in range(len(Y_test)):
    if Y_test[i].argmax() == a2_test[i].argmax():
        true_positive += 1
        
# print(true_positive)
accuracy = true_positive/len(X_test)
print("Accuracy of Model: ", accuracy*100)
classes = ['top','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
print('\nClasification report:\n', classification_report(Y_test.argmax(axis=1), a2_test.argmax(axis=1), target_names=classes))
print('\nConfussion matrix:\n',confusion_matrix(Y_test.argmax(axis=1), a2_test.argmax(axis=1)))


# End of TASK 1

# In[105]:


plt.figure()
plt.imshow(X_train[0].reshape(28,28))
plt.colorbar()
plt.grid(False)
plt.show()


# In[112]:


class_names = ['top','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
plt.figure()
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i].reshape(28,28), cmap=plt.cm.binary)
plt.show()

