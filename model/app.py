from flask import Flask, jsonify, request
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Define the LinearRegression model class
class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# Load the trained model
model = LinearRegression(input_dim=16)
model.load_state_dict(torch.load('linear_model.pth'))
model.eval()

# Create a StandardScaler object for preprocessing input data
scaler = StandardScaler()

# Define a route for prediction
@app.route('/flask/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            if request.is_json:
                # If the content type is JSON, read data accordingly
                input_data = request.json['data']
            else:
                # If the content type is form data, read data accordingly
                input_data = request.form['inputData']
                input_data = list(map(float, input_data.split(',')))

            # Preprocess the input data
            input_data_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
            input_tensor = torch.FloatTensor(input_data_scaled)

            # Make prediction
            with torch.no_grad():
                prediction = model(input_tensor)

            # Convert the prediction to a Python float
            prediction = prediction.item()

            # Return the prediction as JSON
            return jsonify({'prediction': prediction})
        else:
            # Handle GET requests (if needed)
            return jsonify({'message': 'GET request received'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
