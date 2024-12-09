#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[2]:


df = pd.read_csv("Final_News_DF_Labeled_ExamDataset.csv")


# In[3]:


df.columns = [i.lower() for i in df.columns]


# In[4]:


df.head()


# In[5]:


df["label"].unique()


# In[6]:


lencoder = LabelEncoder()
df["label"] = lencoder.fit_transform(df["label"])


# In[7]:


df["label"].unique()


# In[8]:


df["agency"] = df["agency"].astype("int32")


# In[9]:


df["agency"].dtype


# In[10]:


for i in df.columns:
    df[i] = df[i].astype("int32")


# In[11]:


df.info()


# In[12]:


df.head()


# In[13]:


x_train, x_test , y_train , y_test = train_test_split(df.iloc[:,1:].values,df["label"].values,test_size=0.2,random_state=10)


# In[14]:


x_train.shape, x_test.shape , y_train.shape , y_test.shape


# In[15]:


def model_fit_predict(model , x_train, x_test , y_train , y_test):
    history  = model.fit(x_train, y_train, epochs=5, batch_size=32)
    epochs_loss = history.history['loss']
    epochs_accuracy = history.history["accuracy"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2))
    ax1.plot(epochs_loss)
    ax1.set_title('Loss over epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax2.plot(epochs_accuracy)
    ax2.set_title('Accuracy over epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()
    
    y_predicted = model.predict(x_test)
    print("y_predicted :",y_predicted[0])
    #print(("x_test :",x_test[0]))
    y_predicted_labels = [np.argmax(i) for i in y_predicted]
    
    cm = confusion_matrix(y_test, y_predicted_labels)
    print("Confusion Matrix:\n", cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    print("accuracy_score :",accuracy_score(y_test,y_predicted_labels))
    print("classification_report \n" , classification_report(y_test,y_predicted_labels))


# In[ ]:





# In[16]:


# ANN Model
model1 = keras.Sequential([
    keras.layers.Input(shape=(300,)),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dropout(0.3), 
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(3,activation="softmax")
])


# In[17]:


# LSTM Model
model2 = keras.Sequential([
    keras.layers.Input(shape=(300, 1)),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.LSTM(64),
    keras.layers.Dense(3, activation='sigmoid')
])


# In[18]:


# CNN Model
model3 = keras.Sequential([
    keras.layers.Input(shape=(300, 1)),
    keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])


# In[19]:


model1.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# In[20]:


model2.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# In[21]:


model3.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# In[22]:


model_fit_predict(model1 , x_train, x_test , y_train , y_test)


# In[23]:


model_fit_predict(model2 , x_train.reshape(-1, 300, 1), x_test.reshape(-1, 300, 1) , y_train , y_test)


# In[24]:


model_fit_predict(model3 , x_train.reshape(-1, 300, 1), x_test.reshape(-1, 300, 1) , y_train , y_test)


# In[ ]:





# In[ ]:




