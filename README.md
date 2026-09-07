# Universal Forecasting Pipeline

A Streamlit-based forecasting application that supports both Machine Learning and Time Series models for predicting future volumes.

## Features

- **Data Upload**: Support for CSV and Excel files
- **Multiple Models**: 
  - ML: Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM
  - Time Series: ARIMA, SARIMA, Holt-Winters, Prophet
- **Auto Model Selection**: Automatically selects the best model based on MAPE
- **Interactive Visualizations**: Plotly charts for model comparison and forecasts
- **Export Results**: Download forecast as CSV

## Local Development

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run App.py
```

## Deploy to Render

1. Push this code to a GitHub repository

2. Go to [Render Dashboard](https://dashboard.render.com/)

3. Click "New" > "Web Service"

4. Connect your GitHub repository

5. Configure:
   - **Name**: forecasting-pipeline
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run App.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

6. Click "Create Web Service"

## Data Format

Your CSV/Excel should have at minimum:
- A **Date** column (any date format)
- A **Volume/Target** column (numeric values to predict)

Optional columns for better predictions:
- AHT (Average Handle Time)
- Call_Type / Sales_Type
- City
- Language
- Channel

## Sample Data

A `sample_data.csv` is included for testing the application.
