import requests
import json

url = 'http://127.0.0.1:5000/predict'

data = int(input("What month value would you like to predict?: "))

sample_data = {
    'month_value': data 
}

response = requests.post(url, json=sample_data)

if response.status_code == 200:
    try:
        print(f"Prediction: {json.loads(response.text)['prediction']}")
    except KeyError:
        print(f"Failed to get prediction. Response: {response.text}")
else:
    print(f"Failed to get prediction. Status Code: {response.status_code}, Response: {response.text}")
