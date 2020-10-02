from sample_data import data_input
import requests
#Url for prediction
URL = 'http://127.0.0.1:5000/predict'
#headers
headers = {'Content-Type':'application/json'}
#data input / features based on which the prediction is made
data = {'input':data_input}
#getting prediction
r = requests.get(URL,headers = headers,json=data)
#rendering prediction on screen
print('$'+str(round(r.json()['response'],2))+'K')