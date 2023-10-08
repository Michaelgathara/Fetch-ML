from flask import Flask, request, jsonify
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn as nn
import joblib


app = Flask(__name__)

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNN, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

scaler = joblib.load('models/scaler.pkl')
model = FeedForwardNN(1, 10, 1)
model.load_state_dict(torch.load('models/predictor.pt'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    """
        Make a prediction based on the received data and return it as a JSON object.
        
        The function expects a JSON object in the request payload with a key 'month_value' that holds
        the number of scanned receipts for a specific month. The value should be a float or integer.
        The function performs the following steps:
        
        1. Parse the incoming JSON payload to extract the 'month_value'.
        2. Normalize the 'month_value' using a pre-fitted MinMaxScaler.
        3. Convert the normalized value to a PyTorch tensor.
        4. Use the pre-trained PyTorch model to make a prediction.
        5. Rescale the prediction back to the original value range.
        6. Return the prediction as a JSON object with a key 'prediction'.
        
        If any error occurs during these steps, an error message is returned as a JSON object
        with a key 'error'.
        
        Returns:
        --------
        JSON object: Either contains the prediction with key 'prediction' or an error message with key 'error'.
        
        Example:
        --------
        Request payload: {'month_value': 7500000}
        Successful Response: {'prediction': 7491629.5}
        Error Response: {'error': 'Some error message'}
    """
    try:
        data = request.get_json()
        monthValue = float(data['month_value'])
        
        monthValueNormalized = scaler.transform(np.array([[monthValue]]))
        
        monthTensor = torch.FloatTensor(monthValueNormalized)
        with torch.no_grad():
            prediction = model(monthTensor)
        
        prediction_rescaled = scaler.inverse_transform(prediction.numpy())
        
        return jsonify({'prediction': float(prediction_rescaled[0][0])})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
