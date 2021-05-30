import streamlit as st
from annotated_text import annotated_text, annotation
import streamlit.components.v1 as components
import numpy as np
import pickle
import cv2
import os
#import matplotlib.pyplot as plt
import tensorflow as tf
from os import listdir
from tensorflow import keras
#from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
#from sklearn.preprocessing import MultiLabelBinarizer
#from sklearn.model_selection import train_test_split
from PIL import Image
from getarray import resultIdx
from getarray import unhealthy
from disease_description import disease_dic



def modelarch():
  model=Sequential()
  model.add(Conv2D(32, (3, 3),   padding="same",input_shape=(64,64,3)))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(3, 3)))
  model.add(Dropout(0.25))
  model.add(Conv2D(64, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(Conv2D(128, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Conv2D(256, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(Conv2D(512, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(1024))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(Dense(39))
  model.add(Activation("softmax"))
  model.compile(loss="binary_crossentropy",   optimizer='adam', metrics=["accuracy"])
  
  return model



st.header("nXGenAgri")
st.title('Welcome to project nxGenAgri')
st.write("""
Created by Team: The Invincibles.
""")

#Load in model
tempModel = pickle.load(open('finalized_model_weights.pkl', 'rb'))
nxModel = modelarch()
nxModel.set_weights(tempModel)


fl  = st.file_uploader('Upload an image')
if fl:
    img= Image.open(fl)
    #img = cv2.resize(cv2.imread(imgx),(64,64))
    resized_image= img.resize((256,256))
    img = img.resize((64,64))
    img= np.asarray(img)
    img=img.reshape(1,64,64,3) 
    print(img)
    st.title('Here is the image you have uploaded') 
    st.image(resized_image)
    prediction = nxModel.predict_classes(img)
    print("the prediction")
    print(prediction)
    #st.write(prediction)
    end_res = resultIdx[int(prediction)] 
    if int(prediction) in unhealthy:
      annotated_text( (end_res,"","#faa") )
    else:
      annotated_text((end_res,"","#afa"))
    html_string = disease_dic[end_res] 
    components.html(
      html_string,
      height=500  
    )
    #print(prediction)
else:
    st.write('Sorry no image uploaded')


     

# Apply model to make predictions
#prediction = nxModel.predict_classes(fl)
#prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
st.write("""
  Our model will predict which plant have been affected by disease!!!
""")


