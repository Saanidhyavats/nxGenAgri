# Importing essential libraries and modules
from types import MethodDescriptorType
from flask import Flask, render_template, request, Markup
import numpy as np
import json
import pickle
import os
import threading
import time
import multiprocessing
import requests
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
from PIL import Image
from getarray import resultIdx
from getarray import unhealthy
from disease_description import disease_dic
import croprecommend

# loading the model
print('opening model')
nxModel = keras.models.load_model('./mymodel.h5')
print('model loaded')


# =========================================================================================

# Custom functions for calculations




# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@app.route('/')
def home():
    title = 'Harvestify - Home'
    #return render_template('index1.html', title=title)
    return render_template('index1.html')

# render crop recommendation form page


@app.route('/crop',methods=['GET', 'POST'])
def crop_recommend():
    if request.method == 'POST':
        #formData = request.form
        #print(formData)
        #city = formData['adhaar']
        #print(formData['adhaar'])
        #crop = croprecommend.crop_recommend(city)
        #print(crop)
        mimtype = request.mimetype
        print(mimtype)
        request_datajson = request.get_json()
        #request_data = request.get_data()
        #print(request_data)
        print(request_datajson)
        #print('city = ')
        city = request_datajson['city']
        crop = croprecommend.crop_recommend(int(city))
        #print(crop)
        #print(request_datajson['city'])
        #print('my city')
        return crop
    return render_template('crop.html')

# render fertilizer recommendation form page


@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'

    return render_template('mytemplate.html', title=title)

@app.route('/senddata')
def justtry():
    return render_template('justtryhtml.html')

@app.route('/sendjson' , methods=['GET','POST'])
def returnjson():
    #myurl = request.url
    #print(myurl)
    mimtype = request.mimetype
    print(mimtype)
    request_datajson = request.get_json()
    #request_data = request.get_data()
    #print(request_data)
    print(request_datajson)
    # return txid
    return "success"

@app.route('/block', methods=['GET','POST'])
def returnblock():
    if request.method == 'POST':
        mimtype = request.mimetype
        print(mimtype)
        request_datajson = request.get_json()
        print(request_datajson)
        newurl = "http://localhost:3000"
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        r = requests.post(newurl, data=json.dumps(request_datajson), headers=headers)
        #print(r)
        print(r.json()) 
        #print(crop)
        #print(request_datajson['city'])
        #print('my city')
        return r.json()
    return render_template('blockchain1.html')



    

@app.route('/status')
def status():
    return "running"

#disease prediction
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return "nofile"
            #return redirect(request.url)
        #file = request.files['file'].read()
        file = request.files['file']
        if not file:
            return "image not received"
        try:
            print('y')
            print('h')
            img = Image.open(request.files['file'])
            image = img
            if image:
                print('yes')
            else:
                print('no')
            print('myimage')
            print(image)
            image = img.resize((64,64))
            image = np.asarray(image)
            image = image.reshape(1,64,64,3)
            print(image)
            prediction = np.argmax(nxModel.predict(image), axis=-1) 
            print(prediction)
            end_res = resultIdx[int(prediction)] 
            print(end_res)
            html_string = disease_dic[end_res] 
            html_string = Markup(str(html_string))
            return render_template('diseaseresult.html', prediction=html_string, title=title)
        except:
            pass
    return render_template('mytemplate.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False, host='localhost',port=8000)    


    
