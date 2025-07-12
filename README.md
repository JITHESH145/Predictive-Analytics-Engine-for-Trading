# Stock Market Price Prediction

This project is a comprehensive machine learning application that predicts the next day's closing price of a stock. It includes scripts for data preparation, model training, evaluation, and a user-friendly web interface built with Streamlit.

---

## Features

- **Data Preparation**: Downloads historical stock data using `yfinance`.
- **Feature Engineering**: Creates a 'Next_Close' target variable.
- **Data Scaling**: Uses `StandardScaler` to normalize features for optimal model performance.
- **Multiple Models**: Trains and evaluates several regression models, including Linear Regression, Ridge, Lasso, Random Forest, and SVR.
- **Hyperparameter Tuning**: Includes basic tuning for Ridge, Lasso, and Random Forest models.
- **Interactive UI**: A Streamlit application allows users to:
    - Select from any of the trained models.
    - Input any stock symbol.
    - Define a date range for prediction.
    - Visualize predictions against actual prices.
    - Get predictions for hypothetical, user-defined data.

---

## How to Run This Project

### Prerequisites

- Python 3.7+
- Pip (Python package installer)

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to keep project dependencies isolated.

**On Windows:**
```bash
python -m venv env
env\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies

Install all the required libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Run the Machine Learning Pipeline

You must run the scripts in the following order. The first two scripts generate the necessary data and model files that the application uses.

**Step 1: Prepare the Data**
This script downloads the data and creates the scaled data files (`.pkl`) and the `scaler.pkl`.
```bash
python data_preparation.py
```

**Step 2: Train the Models**
This script trains all the models and saves them as `.pkl` files.
```bash
python model_training.py
```

**Step 3 (Optional): Evaluate the Models**
This script provides a performance comparison of all trained models.
```bash
python model_evaluation.py
```

### 5. Launch the Streamlit Application

Now you can run the interactive web application.

```bash
streamlit run app.py
```

This will open the application in your default web browser.
