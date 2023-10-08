# Fetch-ML
This project contains a machine learning model that predicts the number of receipts scanned in a month based on historical data. The model is implemented using PyTorch and is served via a Flask web application.

## Table of Contents
1. [Fast Setup](#fast-setup)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Training the model](#training-the-model)
5. [Testing the API](#testing-the-api)

## Fast Setup
You can setup this project quickly using Docker. To do this, 
1. Download Docker [here](https://www.docker.com/products/docker-desktop/)
2. Start the Docker Engine after install
3. Build and run the image
```sh
docker build -t predictor .
docker run -p 5000:5000 predictor
```
This will make the flask app accessible at `https://localhost:5000`
Move to the [Testing the API](#testing-the-api) section

## Manual Setup
### Requirements
Python, downloadble [here](https://python.org)
Git, downloadble [here](https://git-scm.com/downloads)

### Installation
1. Clone the repo
```sh
git clone https://github.com/Michaelgathara/Fetch-ML
cd Fetch-ML
```

2. Install the required packages:
```sh
pip install -r requirements.txt
```

## Training the model
1. Open `training.ipynb` within the `notebooks/` folder
2. Choose your Juypter Kernel, you should choose the latest Python version you installed above and it will install all the neccesary things for you
3. Run the entire notebook from top to bottom

The training section of the model will output the trained model and a MinMaxScaler to the `app/models/` folder

## Testing the API
You may use curl or Postman to test the model

```sh
curl -X POST -H "Content-Type: application/json" -d '{"month_value": 7500000}' http://localhost:5000/predict
```
