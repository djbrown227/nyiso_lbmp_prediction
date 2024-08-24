# NYISO Price Prediction API

This project is part of my data science portfolio and demonstrates a complete machine learning pipeline for predicting NYISO (New York Independent System Operator) electricity prices using a combination of Python, FastAPI, and Rust with Actix-Web. The project consists of a Python service for training and serving the model and a Rust service that interacts with the Python API.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Model Training](#model-training)
- [API Endpoints](#api-endpoints)
- [Running the Project](#running-the-project)
- [Improvements](#improvements)
- [License](#license)

## Project Overview

The main goal of this project is to predict the Locational Based Marginal Price (LBMP) of electricity in the NYISO market based on the marginal cost of losses and congestion. The project includes the following components:

1. **Model Training:** A Python script that trains a Random Forest Regressor using NYISO generator price data.
2. **Prediction API:** A FastAPI-based Python service that serves the trained model to make real-time predictions.
3. **Rust Integration:** A Rust service using Actix-Web to call the Python API and expose its own endpoint for predictions.

## Data

The data used for training the model is sourced from an SQL database containing historical NYISO generator prices. The key features include:

- **LBMP ($/MWHr):** The Locational Based Marginal Price, the target variable for prediction.
- **Marginal Cost Losses ($/MWHr):** The cost associated with energy losses during transmission.
- **Marginal Cost Congestion ($/MWHr):** The cost associated with congestion in the transmission network.

## Model Training

The model training process is encapsulated in the `train_model.py` script. The script performs the following steps:

1. **Data Extraction:** Retrieves data from a MySQL database.
2. **Data Preprocessing:** Cleans the data and selects the relevant features.
3. **Model Training:** Trains a Random Forest Regressor using `RandomizedSearchCV` to optimize hyperparameters.
4. **Model Evaluation:** Evaluates the model using Mean Squared Error (MSE) on a test set.
5. **Model Saving:** Saves the best model to a file using `joblib`.

## API Endpoints

The project exposes two main endpoints:

1. **Python API (FastAPI):**
   - `GET /`: Returns a welcome message.
   - `POST /predict/`: Takes a list of inputs and returns predicted LBMP values.

2. **Rust API (Actix-Web):**
   - `POST /predict`: Calls the Python API's `/predict/` endpoint and returns the response.

### Example Request (Python API)

Here's the content formatted and improved in Markdown:

```markdown
## API Endpoint

### POST /predict/
Example input:
```json
[
    {
        "lbmp": 50.5,
        "marginal_cost_losses": 5.2,
        "marginal_cost_congestion": 3.8
    }
]
```

## Running the Project

### Prerequisites
- Python 3.x
- Rust and Cargo
- FastAPI and Uvicorn
- MySQL database with NYISO data

### Steps

#### Train the Model
```bash
python train_model.py
```

#### Run the Python API
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

#### Run the Rust API
```bash
cargo run
```

## Make Predictions
Send a POST request to `http://127.0.0.1:8080/predict` with the input data.

## Improvements

There are several ways to improve this project:

- **Data Enrichment**: Include additional features such as weather data, demand forecasts, and economic indicators to improve the model's accuracy.
- **Modeling**: Experiment with other machine learning models like XGBoost, Gradient Boosting Machines, or even deep learning models.
- **Deployment**: Deploy the services using Docker for easier setup and scalability. Consider deploying the services to a cloud provider like AWS or GCP.
```

This version is well-organized and easy to read. You can copy and paste it into your `README.md` file.
