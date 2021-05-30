from typing import get_origin
import requests
import json

url = "http://localhost:3000"  

#data = {'sender': 'Alice', 'receiver': 'Bob', 'message':'We did it!'}

data = {
    "farmer" : "Saanidhya",
    "crop" : "tomato",
    "price" : "22",
    "qty"   : 23,
    "corporate" : "reliance farm"
}

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

r = requests.post(url, data=json.dumps(data), headers=headers)
#print(r)
print(r.json()) 
#mydata = r.json() 
#print(mydata[0])