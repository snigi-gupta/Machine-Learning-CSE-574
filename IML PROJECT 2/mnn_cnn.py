#!/usr/bin/env python
# coding: utf-8

# In[18]:


import matplotlib.pyplot as plt
import util_mnist_reader as mnist_reader
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop,Adadelta,Adam
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.metrics import classification_report, confusion_matrix


# In[19]:


# function to print shape
def print_shape(matrix=None,name=None):
    if type(matrix) is list:
        for m in range(len(matrix)):
            print("Shape of {}:        \t".format(name[m]),matrix[m].shape)
    else:
        print("Shape of {}:        \t".format(name),matrix.shape)
    
    print("------------------------------------------------------------------------")

    
X_train, Y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, Y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')


X_train, X_valid = X_train[5000:], X_train[:5000]
Y_train, Y_valid = Y_train[5000:], Y_train[:5000]

print("Y_train",Y_train.shape)
print("Y_valid",Y_valid.shape)
print("Y_test",Y_test.shape)


# normalize the data
X_train = X_train/255
X_valid = X_valid/255
X_test = X_test/255
#one-hot encode target column
Y_train = to_categorical(Y_train)
Y_valid = to_categorical(Y_valid)
Y_test = to_categorical(Y_test)


# In[20]:


# TASK 2

# define the keras model
multimodel = Sequential()
multimodel.add(Dense(128, activation='sigmoid'))
multimodel.add(Dense(128, activation='sigmoid'))
multimodel.add(Dense(128, input_dim=X_train.shape[1], activation='sigmoid'))
multimodel.add(Dense(Y_train.shape[1], activation='softmax'))

multimodel.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['categorical_accuracy'])
multimodel.fit(X_train, Y_train, epochs=150, batch_size=100, validation_data = (X_valid,Y_valid))


# In[21]:


multimodel.summary()


# In[22]:


plt.figure()
plt.plot(multimodel.history.history['loss'], '-g', label='Training Loss Track')
plt.plot(multimodel.history.history['val_loss'], '-b', label='Validation Loss Track')
plt.legend(loc='upper right')
plt.title('Loss vs Epochs')
print("------------------------------------------------------------------------")
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()

plt.figure()
plt.plot(multimodel.history.history['categorical_accuracy'], '-g', label='Training Accuracy Track')
plt.plot(multimodel.history.history['val_categorical_accuracy'], '-b', label='Validation Accuracy Track')
plt.legend(loc='lower right')
plt.title('Accuracy vs Epochs')
print("------------------------------------------------------------------------")
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()


# In[24]:


Y_prediction = multimodel.predict(X_test)
classes = ['top','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
#multimodel_accuracy = multimodel.evaluate(X_test,Y_test)
#print(multimodel_accuracy[1])
print('\nClasification report:\n', classification_report(Y_test.argmax(axis=1), Y_prediction.argmax(axis=1), target_names=classes))
print('\nConfussion matrix:\n',confusion_matrix(Y_test.argmax(axis=1), Y_prediction.argmax(axis=1)))


# In[12]:


# TASK 3

# reshaping in order to fit the model. Here paramters are reshape(X_train_len,image_row_len,image_col_len,1). 
# 1 is for grayscale image. Image size is 28x28 (flattened image is 784)
X_train = X_train.reshape(-1,28,28,1)
X_valid = X_valid.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

# createing the cnn model
cnn_model = Sequential()

cnn_model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1), padding='same'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
cnn_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
cnn_model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=20)


# In[13]:


cnn_model.summary()


# In[14]:


Y_prediction = cnn_model.predict(X_test)

class_names = ['top','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
print('\nClasification report:\n', classification_report(Y_test.argmax(axis=1), Y_prediction.argmax(axis=1), target_names=class_names))
print('\nConfussion matrix:\n',confusion_matrix(Y_test.argmax(axis=1), Y_prediction.argmax(axis=1)))


# In[16]:


plt.figure()
plt.plot(cnn_model.history.history['loss'], '-g', label='Training Loss Track')
plt.plot(cnn_model.history.history['val_loss'], '-b', label='Validation Loss Track')
plt.legend(loc='upper right')
plt.title('Loss vs Epochs')
print("------------------------------------------------------------------------")
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()

plt.figure()
plt.plot(cnn_model.history.history['accuracy'], '-g', label='Training Accuracy Track')
plt.plot(cnn_model.history.history['val_accuracy'], '-b', label='Validation Accuracy Track')
plt.legend(loc='lower right')
print("------------------------------------------------------------------------")
print("Accuracy vs Epochs")
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()


# In[ ]:




