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
    TODO: Add docs
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
