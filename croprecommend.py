import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
load=joblib.load('./models/randomforest.joblib')
#URL = "http://api.openweathermap.org/data/2.5/forecast?q=nainital&appid=c899bb1f9b0daede86ae06cde29465d8"
# -----------N,P,K,temp,humidity,Ph,precipitation------


"""l=[1,2,3,4,5,6,200]
print(l)
l=np.array(l)
print(l)
l=l.reshape(1,7)
print(l)
print(load.predict(l))
"""



x = pd.read_csv('./utils/samplecrop.csv')
def crop_recommend(no):
    city = x.loc[no][5]
    print("no = " + str(no) )
    print("city = " + city)
    URL = "http://api.openweathermap.org/data/2.5/forecast?q=" + city + "&appid=c899bb1f9b0daede86ae06cde29465d8"
    data = requests.get(url=URL)
    data = data.json()
    data1 = data['list'][0]
    temp = 0
    cnt = 0
    for elem in data['list']:
        temp+=elem['main']['temp']
        cnt+=1
    temp = temp/cnt
    humidity = 0
    for elem in data['list']:
        humidity+=elem['main']['humidity']
    humidity = humidity/cnt
    rain=0
    cnt=0
    for elem in data['list']:
        if 'rain' in elem.keys():
            rain+=elem['rain']['3h']
            cnt+=1
    if rain>0:
        rain = rain/cnt    
    
    #if 'rain' in data1.keys():
    #    rain = data1['rain']['3h']
    #else:
    #    rain = 0    
    #rain = data1['rain']['3h']
    city_details = data['city']
    #print("debug1")
    #print(data1)
    #print("success")
    #print(city_details)
    #print("temp = " + str(temp) + " rain = " + str(rain) + " humidity = " + str(humidity) )
    l = []
    #print(x) 
    #print(x.loc[0][1])
    l.append(x.loc[no][1])
    l.append(x.loc[no][2])
    l.append(x.loc[no][3])
    #l.append(x.loc[0][4])
    #print(l)
    l.append(temp)
    l.append(humidity)
    l.append(x.loc[no][4])
    l.append(rain)
    #print(l)
    l1 = np.array(l)
    #print(l1)
    l1 = np.array([l1])
    print(l1)
    predict = load.predict(l1)
    #print(predict)
    print(predict[0])
    return predict[0]

#crop_recommend(3)   

    


