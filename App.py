import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import os
import psutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

def get_available_memory_gb():
    """Get available system memory in GB"""
    try:
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return 4.0  # Assume 4GB if can't detect

def is_low_memory_environment():
    """Check if running in a memory-constrained environment"""
    available_gb = get_available_memory_gb()
    return available_gb < 1.5  # Less than 1.5GB available

# Check memory at startup
LOW_MEMORY_MODE = is_low_memory_environment()

st.set_page_config(
    page_title="Forecasting Pipeline",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Universal Forecasting Pipeline")
st.markdown("Upload historical data and predict future volumes using ML, Deep Learning & Time Series models")

# Model descriptions - when and why each model works best
MODEL_DESCRIPTIONS = {
    # Machine Learning Models
    'Linear Regression': 'Use when: Steady growth/decline trends. Example: Monthly sales increasing 5% consistently. Needs: 50+ rows.',
    'Ridge Regression': 'Use when: Many features causing overfitting. Example: 10+ columns like City, Channel, AHT all affecting volume. Needs: 100+ rows.',
    'Random Forest': 'Use when: Complex feature interactions. Example: Volume depends on City AND Channel together, not separately. Needs: 200+ rows.',
    'Gradient Boosting': 'Use when: Accuracy is critical. Example: Budget planning where 1% error matters. Needs: 300+ rows, slower training.',
    'XGBoost': 'Use when: Production deployment needed. Example: Daily automated forecasts. Needs: 500+ rows, handles missing data well.',
    'LightGBM': 'Use when: Very large data or fast training needed. Example: Millions of rows, hourly retraining. Needs: 1000+ rows ideal.',

    # Time Series Models (Auto-tuned using ACF/PACF)
    'ARIMA': 'Auto-tuned using ACF/PACF analysis. Use when: Trend exists but no clear weekly/monthly pattern. Parameters (p,d,q) found automatically via ADF test and information criteria.',
    'SARIMA': 'Auto-tuned using ACF/PACF analysis. Use when: Clear repeating patterns exist. Parameters (p,d,q)(P,D,Q,m) optimized automatically for your data seasonality.',
    'Holt-Winters': 'Use when: Both trend AND seasonality present. Example: Retail sales growing yearly with holiday spikes. Needs: 2+ years of data.',
    'Prophet': 'Use when: Business data with gaps/holidays. Example: Company closed weekends, Black Friday spikes. Handles missing dates automatically.',

    # Deep Learning Models
    'LSTM': 'Use when: Long-term memory matters (30+ days). Example: Volume today depends on what happened 2 months ago. Needs: 500+ rows.',
    'GRU': 'Use when: Similar to LSTM but need faster training. Example: Real-time forecasting systems. Needs: 500+ rows, 30% faster than LSTM.',
    '1D-CNN': 'Use when: Local short-term patterns dominate. Example: Sudden spikes matter more than gradual trends. Needs: 300+ rows.',
    'Dense NN': 'Use when: Unsure which DL model fits. Example: First deep learning attempt on your data. Needs: 200+ rows, good baseline.',
    'Transformer': 'Use when: Some dates matter more than others. Example: Promo days heavily influence next week, normal days less so. Needs: 500+ rows.',
    'NeuralProphet': 'Use when: You want Prophet + neural network combined. Automatically learns trend, seasonality, AND complex patterns. Example: Call center with holidays, special events, and unpredictable surges. Needs: 300+ rows.',

    # Ensemble
    'Ensemble (Top 3)': 'Weighted average of top 3 ML models by MAPE. Use when: You want maximum stability and reduced variance. Combines strengths of multiple models. Often outperforms any single model.',

    # Phase 2 Models
    'Bidirectional LSTM': 'Reads data forward AND backward. Use when: Patterns exist in both directions. Example: Weekly patterns where Tuesday depends on both Monday AND Wednesday. 10-15% better than regular LSTM.',
}


def get_model_description(model_name):
    """Get description for a model, handling partial matches and dynamic names"""
    if model_name in MODEL_DESCRIPTIONS:
        return MODEL_DESCRIPTIONS[model_name]

    # Handle dynamic ARIMA/SARIMA names like ARIMA(1,1,0) or SARIMA(1,0,1)(1,1,1,7)
    if model_name.startswith('ARIMA') and 'SARIMA' not in model_name:
        return MODEL_DESCRIPTIONS.get('ARIMA', 'Auto-tuned ARIMA model with optimal (p,d,q) parameters.')
    if model_name.startswith('SARIMA'):
        return MODEL_DESCRIPTIONS.get('SARIMA', 'Auto-tuned SARIMA model with optimal seasonal parameters.')

    for key in MODEL_DESCRIPTIONS:
        if key in model_name or model_name in key:
            return MODEL_DESCRIPTIONS[key]
    return 'No description available'


def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-08-31', freq='D')
    n = len(dates)

    data = {
        'Date': dates,
        'Volume': np.random.poisson(500, n) + np.sin(np.arange(n) * 2 * np.pi / 365) * 100,
        'AHT': np.random.normal(300, 50, n),
        'Call_Type': np.random.choice(['Inbound', 'Outbound', 'Transfer'], n),
        'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], n),
        'Language': np.random.choice(['English', 'Spanish', 'French'], n),
        'Channel': np.random.choice(['Phone', 'Chat', 'Email'], n)
    }
    return pd.DataFrame(data)


def preprocess_data(df, date_col, target_col):
    """Preprocess the data for modeling"""
    df = df.copy()

    # Safe date parsing with error handling
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if df[date_col].isna().all():
            raise ValueError(f"Could not parse any dates in column '{date_col}'")
        # Drop rows with unparseable dates
        invalid_dates = df[date_col].isna().sum()
        if invalid_dates > 0:
            df = df.dropna(subset=[date_col])
    except Exception as e:
        raise ValueError(f"Error parsing dates: {str(e)}")

    df = df.sort_values(date_col).reset_index(drop=True)

    df['year'] = df[date_col].dt.year.astype('float64')
    df['month'] = df[date_col].dt.month.astype('float64')
    df['day'] = df[date_col].dt.day.astype('float64')
    df['dayofweek'] = df[date_col].dt.dayofweek.astype('float64')
    df['dayofyear'] = df[date_col].dt.dayofyear.astype('float64')
    df['weekofyear'] = df[date_col].dt.isocalendar().week.astype('float64')
    df['quarter'] = df[date_col].dt.quarter.astype('float64')
    df['is_weekend'] = (df['dayofweek'] >= 5).astype('float64')
    df['is_month_start'] = df[date_col].dt.is_month_start.astype('float64')
    df['is_month_end'] = df[date_col].dt.is_month_end.astype('float64')

    return df


def add_lag_features(df, target_col, lags=[1, 2, 3, 7, 14, 30]):
    """Add lag features - previous day/week/month values"""
    df = df.copy()
    for lag in lags:
        if len(df) > lag:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
    return df


def add_rolling_features(df, target_col, windows=[7, 14, 30]):
    """Add rolling statistics - moving average, std, min, max"""
    df = df.copy()
    for window in windows:
        if len(df) > window:
            df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
            df[f'rolling_min_{window}'] = df[target_col].shift(1).rolling(window=window).min()
            df[f'rolling_max_{window}'] = df[target_col].shift(1).rolling(window=window).max()
    return df


def detect_and_handle_outliers(df, target_col, method='iqr', threshold=1.5):
    """Detect and handle outliers using IQR or Z-score method"""
    df = df.copy()
    original_len = len(df)

    if method == 'iqr':
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_mask = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
    else:  # z-score
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[target_col].dropna()))
        outlier_mask = z_scores > threshold
        # Calculate bounds for z-score method
        mean_val = df[target_col].mean()
        std_val = df[target_col].std()
        # Handle zero std (all same values) - no outliers possible
        if std_val == 0 or pd.isna(std_val):
            return df, 0
        lower_bound = mean_val - threshold * std_val
        upper_bound = mean_val + threshold * std_val

    outlier_count = outlier_mask.sum()

    # Cap outliers instead of removing (preserves time series continuity)
    if outlier_count > 0:
        df.loc[df[target_col] < lower_bound, target_col] = lower_bound
        df.loc[df[target_col] > upper_bound, target_col] = upper_bound

    return df, outlier_count


def create_ensemble_prediction(models_dict, X_test, results_df, top_n=3):
    """Create weighted ensemble from top N models based on MAPE"""
    # Get top N models by MAPE
    top_models = results_df.nsmallest(top_n, 'MAPE (%)')

    predictions = []
    weights = []

    for _, row in top_models.iterrows():
        model_name = row['Model']
        mape = row['MAPE (%)']

        if model_name in models_dict:
            model = models_dict[model_name]
            try:
                pred = model.predict(X_test)
                predictions.append(pred)
                # Weight by inverse MAPE (lower error = higher weight)
                weights.append(1.0 / (mape + 0.01))
            except Exception:
                pass

    if not predictions:
        return None

    # Normalize weights
    weight_sum = sum(weights)
    if weight_sum == 0:
        return None
    weights = np.array(weights) / weight_sum

    # Weighted average
    ensemble_pred = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        ensemble_pred += pred * weight

    return ensemble_pred


def calculate_prediction_intervals(model, X, scaler, n_iterations=100, confidence=0.95):
    """
    Calculate prediction intervals using Monte Carlo Dropout.
    Runs multiple forward passes with dropout enabled to estimate uncertainty.
    """
    import tensorflow as tf

    predictions = []

    # Enable dropout during inference by using training=True
    for _ in range(n_iterations):
        pred = model(X, training=True)  # Keep dropout active
        predictions.append(pred.numpy())

    predictions = np.array(predictions).squeeze()

    # Calculate mean and percentiles
    mean_pred = np.mean(predictions, axis=0)
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100

    lower_bound = np.percentile(predictions, lower_percentile, axis=0)
    upper_bound = np.percentile(predictions, upper_percentile, axis=0)

    # Inverse transform to original scale
    mean_pred = scaler.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
    lower_bound = scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
    upper_bound = scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten()

    return mean_pred, lower_bound, upper_bound


def calculate_shap_values(model, X_train, X_test, feature_names, model_name):
    """
    Calculate SHAP values for model explainability.
    Returns feature importance and individual prediction explanations.
    """
    try:
        import shap

        # Use appropriate explainer based on model type
        if 'XGBoost' in model_name or 'LightGBM' in model_name or 'Random Forest' in model_name or 'Gradient' in model_name:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            # For linear models, use LinearExplainer or KernelExplainer
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test)

        # Calculate feature importance (mean absolute SHAP value)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('Importance', ascending=False)

        return shap_values, feature_importance, explainer
    except Exception as e:
        return None, None, None


def detect_data_drift(train_data, new_data, target_col, threshold=0.05):
    """
    Detect data drift between training data and new data using statistical tests.
    Uses Kolmogorov-Smirnov test for numerical columns.
    Returns drift report with alerts.
    """
    from scipy import stats

    drift_report = []
    has_drift = False

    # Check target variable drift
    stat, p_value = stats.ks_2samp(train_data[target_col], new_data[target_col])
    drift_detected = p_value < threshold
    drift_report.append({
        'Column': target_col,
        'Test': 'KS Test',
        'Statistic': round(stat, 4),
        'P-Value': round(p_value, 4),
        'Drift Detected': '⚠️ YES' if drift_detected else '✅ NO'
    })
    if drift_detected:
        has_drift = True

    # Check numerical columns
    numerical_cols = train_data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col != target_col and col in new_data.columns:
            try:
                stat, p_value = stats.ks_2samp(train_data[col].dropna(), new_data[col].dropna())
                drift_detected = p_value < threshold
                drift_report.append({
                    'Column': col,
                    'Test': 'KS Test',
                    'Statistic': round(stat, 4),
                    'P-Value': round(p_value, 4),
                    'Drift Detected': '⚠️ YES' if drift_detected else '✅ NO'
                })
                if drift_detected:
                    has_drift = True
            except Exception:
                pass

    return pd.DataFrame(drift_report), has_drift


def calculate_model_monitoring_metrics(y_true, y_pred, baseline_mape=None):
    """
    Calculate monitoring metrics to detect model degradation.
    Compares current performance against baseline.
    """
    current_mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    current_mae = mean_absolute_error(y_true, y_pred)

    metrics = {
        'Current MAPE (%)': round(current_mape, 2),
        'Current MAE': round(current_mae, 2),
    }

    if baseline_mape and baseline_mape > 0:
        degradation = ((current_mape - baseline_mape) / baseline_mape) * 100
        metrics['Baseline MAPE (%)'] = round(baseline_mape, 2)
        metrics['Degradation (%)'] = round(degradation, 2)
        metrics['Status'] = '⚠️ RETRAIN NEEDED' if degradation > 20 else '✅ HEALTHY'
    elif baseline_mape == 0:
        metrics['Baseline MAPE (%)'] = 0
        metrics['Status'] = '✅ HEALTHY' if current_mape < 5 else '⚠️ CHECK MODEL'

    return metrics


def encode_categorical(df, categorical_cols):
    """Encode categorical columns"""
    df = df.copy()
    encoders = {}

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str)).astype('float64')
            encoders[col] = le

    return df, encoders


def get_ml_features(df, target_col, date_col, categorical_cols):
    """Get feature columns for ML models"""
    date_features = ['year', 'month', 'day', 'dayofweek', 'dayofyear',
                     'weekofyear', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end']

    encoded_cols = [col + '_encoded' for col in categorical_cols if col + '_encoded' in df.columns]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in [target_col] + date_features + encoded_cols]

    feature_cols = date_features + encoded_cols + numeric_cols
    feature_cols = [col for col in feature_cols if col in df.columns and col != target_col]

    return feature_cols


def train_ml_models(X_train, y_train, X_test, y_test, use_cv=False, use_tuning=False):
    """Train multiple ML models with optional cross-validation and Optuna hyperparameter tuning"""
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    results = []
    trained_models = {}

    # Time series cross-validation splitter
    tscv = TimeSeriesSplit(n_splits=5) if use_cv else None

    def create_objective(model_name, X, y, cv):
        """Create Optuna objective function for each model type"""
        def objective(trial):
            if model_name == 'Ridge Regression':
                alpha = trial.suggest_float('alpha', 0.001, 100.0, log=True)
                model = Ridge(alpha=alpha)
            elif model_name == 'Random Forest':
                model = RandomForestRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    max_depth=trial.suggest_int('max_depth', 3, 30),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    random_state=42, n_jobs=-1
                )
            elif model_name == 'Gradient Boosting':
                model = GradientBoostingRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    max_depth=trial.suggest_int('max_depth', 3, 15),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    random_state=42
                )
            elif model_name == 'XGBoost':
                model = xgb.XGBRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    max_depth=trial.suggest_int('max_depth', 3, 15),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    random_state=42, verbosity=0
                )
            elif model_name == 'LightGBM':
                model = lgb.LGBMRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    num_leaves=trial.suggest_int('num_leaves', 10, 100),
                    max_depth=trial.suggest_int('max_depth', 3, 15),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    random_state=42, verbose=-1
                )
            else:
                return float('inf')

            if cv:
                scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
                return -scores.mean()
            else:
                model.fit(X, y)
                return mean_absolute_error(y, model.predict(X))

        return objective

    # Models to train
    model_configs = [
        ('Linear Regression', LinearRegression(), False),
        ('Ridge Regression', Ridge(), True),
        ('Random Forest', RandomForestRegressor(random_state=42, n_jobs=-1), True),
        ('Gradient Boosting', GradientBoostingRegressor(random_state=42), True),
        ('XGBoost', xgb.XGBRegressor(random_state=42, verbosity=0), True),
        ('LightGBM', lgb.LGBMRegressor(random_state=42, verbose=-1), True)
    ]

    for name, default_model, can_tune in model_configs:
        try:
            if use_tuning and can_tune:
                st.text(f"Optuna tuning {name} (20 trials)...")
                study = optuna.create_study(direction='minimize')
                study.optimize(
                    create_objective(name, X_train, y_train, tscv if use_cv else None),
                    n_trials=20,
                    show_progress_bar=False
                )
                best_params = study.best_params
                st.text(f"Best params for {name}: {best_params}")

                # Recreate model with best params
                if name == 'Ridge Regression':
                    final_model = Ridge(**best_params)
                elif name == 'Random Forest':
                    final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
                elif name == 'Gradient Boosting':
                    final_model = GradientBoostingRegressor(**best_params, random_state=42)
                elif name == 'XGBoost':
                    final_model = xgb.XGBRegressor(**best_params, random_state=42, verbosity=0)
                elif name == 'LightGBM':
                    final_model = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)

                final_model.fit(X_train, y_train)
            else:
                final_model = default_model
                final_model.fit(X_train, y_train)

            # Cross-validation scores
            cv_score = None
            if use_cv and not use_tuning:
                cv_scores = cross_val_score(final_model, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
                cv_score = -cv_scores.mean()

            predictions = final_model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mape = mean_absolute_percentage_error(y_test, predictions) * 100

            result = {
                'Model': name,
                'MAE': round(mae, 2),
                'RMSE': round(rmse, 2),
                'MAPE (%)': round(mape, 2),
                'Type': 'Machine Learning'
            }
            if cv_score:
                result['CV MAE'] = round(cv_score, 2)

            results.append(result)
            trained_models[name] = final_model
        except Exception as e:
            st.warning(f"Error training {name}: {str(e)}")

    return results, trained_models


def create_sequences(data, seq_length):
    """Create sequences for LSTM/GRU models"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def train_deep_learning_models(train_series, test_series, seq_length=30, use_tuning=False, low_memory=False):
    """Train deep learning models with production-grade techniques"""
    dl_models = ['LSTM', 'Bidirectional LSTM', 'GRU', '1D-CNN', 'Dense NN', 'Transformer']
    skipped_results = []

    try:
        import tensorflow as tf  # type: ignore
        from tensorflow.keras.models import Sequential  # type: ignore
        from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, Flatten, Dropout, Input, BatchNormalization  # type: ignore
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  # type: ignore
        from tensorflow.keras.optimizers import Adam  # type: ignore
        from tensorflow.keras import backend as K  # type: ignore
        tf.get_logger().setLevel('ERROR')

        # Memory-efficient settings
        tf.config.set_soft_device_placement(True)
        # Limit GPU memory growth if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except ImportError:
        st.warning("TensorFlow not installed. Skipping deep learning models.")
        for model_name in dl_models:
            skipped_results.append({
                'Model': model_name,
                'MAE': '-',
                'RMSE': '-',
                'MAPE (%)': '-',
                'Type': 'Deep Learning',
                'Status': '⚠️ Skipped: TensorFlow not installed'
            })
        return skipped_results, {}

    # Adjust model complexity based on memory
    if low_memory:
        units_large = 32  # Reduced from 64
        units_small = 16  # Reduced from 32
        batch_size = 16   # Reduced from 32
        epochs = 50       # Reduced from 100
    else:
        units_large = 64
        units_small = 32
        batch_size = 32
        epochs = 100

    # Check minimum data length
    min_required = seq_length + 10  # Need at least seq_length + some samples for training
    if len(train_series) < min_required:
        st.warning(f"Deep Learning models require at least {min_required} training samples. You have {len(train_series)}. Skipping DL models.")
        for model_name in dl_models:
            skipped_results.append({
                'Model': model_name,
                'MAE': '-',
                'RMSE': '-',
                'MAPE (%)': '-',
                'Type': 'Deep Learning',
                'Status': f'⚠️ Skipped: Need {min_required}+ rows (have {len(train_series)})'
            })
        return skipped_results, {}

    results = []
    trained_models = {}

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))

    full_data = np.concatenate([train_scaled, scaler.transform(test_series.values.reshape(-1, 1))])

    X_train, y_train = create_sequences(train_scaled.flatten(), seq_length)

    test_start_idx = len(train_scaled) - seq_length
    X_test_list, y_test_list = [], []
    for i in range(len(test_series)):
        idx = test_start_idx + i
        X_test_list.append(full_data[idx:idx + seq_length].flatten())
        y_test_list.append(full_data[idx + seq_length])

    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list).flatten()

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Production-grade callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss' if len(X_train) > 100 else 'loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)  # Learning rate scheduling
    ]

    # Validation split for better generalization
    validation_split = 0.2 if len(X_train) > 100 else 0.0

    # LSTM Model with BatchNorm and LR scheduling
    try:
        K.clear_session()  # Free memory from previous models
        lstm_model = Sequential([
            Input(shape=(seq_length, 1)),
            LSTM(units_large, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(units_small),
            BatchNormalization(),
            Dropout(0.3),
            Dense(units_small // 2, activation='relu'),
            Dense(1)
        ])
        lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0,
                       callbacks=callbacks, validation_split=validation_split)

        lstm_pred_scaled = lstm_model.predict(X_test, verbose=0)
        lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        mae = mean_absolute_error(y_test_actual, lstm_pred)
        rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_pred))
        mape = mean_absolute_percentage_error(y_test_actual, lstm_pred) * 100

        results.append({
            'Model': 'LSTM',
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE (%)': round(mape, 2),
            'Type': 'Deep Learning'
        })
        trained_models['LSTM'] = {'model': lstm_model, 'scaler': scaler, 'seq_length': seq_length}
    except Exception as e:
        st.warning(f"LSTM failed: {str(e)}")

    # Bidirectional LSTM (Phase 2) - reads data both forward and backward
    try:
        K.clear_session()
        from tensorflow.keras.layers import Bidirectional  # type: ignore

        bilstm_model = Sequential([
            Input(shape=(seq_length, 1)),
            Bidirectional(LSTM(units_large, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(units_small)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(units_small // 2, activation='relu'),
            Dense(1)
        ])
        bilstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        bilstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0,
                         callbacks=callbacks, validation_split=validation_split)

        bilstm_pred_scaled = bilstm_model.predict(X_test, verbose=0)
        bilstm_pred = scaler.inverse_transform(bilstm_pred_scaled).flatten()

        mae = mean_absolute_error(y_test_actual, bilstm_pred)
        rmse = np.sqrt(mean_squared_error(y_test_actual, bilstm_pred))
        mape = mean_absolute_percentage_error(y_test_actual, bilstm_pred) * 100

        results.append({
            'Model': 'Bidirectional LSTM',
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE (%)': round(mape, 2),
            'Type': 'Deep Learning'
        })
        trained_models['Bidirectional LSTM'] = {'model': bilstm_model, 'scaler': scaler, 'seq_length': seq_length}
    except Exception as e:
        st.warning(f"Bidirectional LSTM failed: {str(e)}")

    # GRU Model with BatchNorm and LR scheduling
    try:
        K.clear_session()
        gru_model = Sequential([
            Input(shape=(seq_length, 1)),
            GRU(units_large, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            GRU(units_small),
            BatchNormalization(),
            Dropout(0.3),
            Dense(units_small // 2, activation='relu'),
            Dense(1)
        ])
        gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        gru_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0,
                      callbacks=callbacks, validation_split=validation_split)

        gru_pred_scaled = gru_model.predict(X_test, verbose=0)
        gru_pred = scaler.inverse_transform(gru_pred_scaled).flatten()

        mae = mean_absolute_error(y_test_actual, gru_pred)
        rmse = np.sqrt(mean_squared_error(y_test_actual, gru_pred))
        mape = mean_absolute_percentage_error(y_test_actual, gru_pred) * 100

        results.append({
            'Model': 'GRU',
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE (%)': round(mape, 2),
            'Type': 'Deep Learning'
        })
        trained_models['GRU'] = {'model': gru_model, 'scaler': scaler, 'seq_length': seq_length}
    except Exception as e:
        st.warning(f"GRU failed: {str(e)}")

    # 1D CNN Model with BatchNorm
    try:
        K.clear_session()
        cnn_model = Sequential([
            Input(shape=(seq_length, 1)),
            Conv1D(filters=units_large, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            Conv1D(filters=units_small, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            Flatten(),
            Dense(units_large, activation='relu'),
            Dropout(0.3),
            Dense(units_small, activation='relu'),
            Dense(1)
        ])
        cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        cnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0,
                      callbacks=callbacks, validation_split=validation_split)

        cnn_pred_scaled = cnn_model.predict(X_test, verbose=0)
        cnn_pred = scaler.inverse_transform(cnn_pred_scaled).flatten()

        mae = mean_absolute_error(y_test_actual, cnn_pred)
        rmse = np.sqrt(mean_squared_error(y_test_actual, cnn_pred))
        mape = mean_absolute_percentage_error(y_test_actual, cnn_pred) * 100

        results.append({
            'Model': '1D-CNN',
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE (%)': round(mape, 2),
            'Type': 'Deep Learning'
        })
        trained_models['1D-CNN'] = {'model': cnn_model, 'scaler': scaler, 'seq_length': seq_length}
    except Exception as e:
        st.warning(f"1D-CNN failed: {str(e)}")

    # Dense Neural Network with BatchNorm
    try:
        K.clear_session()
        X_train_flat = X_train.reshape((X_train.shape[0], -1))
        X_test_flat = X_test.reshape((X_test.shape[0], -1))

        dense_model = Sequential([
            Input(shape=(seq_length,)),
            Dense(units_large * 2, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(units_large, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(units_small, activation='relu'),
            Dense(1)
        ])
        dense_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        dense_model.fit(X_train_flat, y_train, epochs=epochs, batch_size=batch_size, verbose=0,
                        callbacks=callbacks, validation_split=validation_split)

        dense_pred_scaled = dense_model.predict(X_test_flat, verbose=0)
        dense_pred = scaler.inverse_transform(dense_pred_scaled).flatten()

        mae = mean_absolute_error(y_test_actual, dense_pred)
        rmse = np.sqrt(mean_squared_error(y_test_actual, dense_pred))
        mape = mean_absolute_percentage_error(y_test_actual, dense_pred) * 100

        results.append({
            'Model': 'Dense NN',
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE (%)': round(mape, 2),
            'Type': 'Deep Learning'
        })
        trained_models['Dense NN'] = {'model': dense_model, 'scaler': scaler, 'seq_length': seq_length, 'flat': True}
    except Exception as e:
        st.warning(f"Dense NN failed: {str(e)}")

    # Transformer-based Model
    try:
        K.clear_session()
        from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D  # type: ignore

        inputs = tf.keras.Input(shape=(seq_length, 1))
        x = Dense(units_small)(inputs)
        attn_output = MultiHeadAttention(num_heads=2, key_dim=units_small)(x, x)
        x = LayerNormalization()(x + attn_output)
        x = GlobalAveragePooling1D()(x)
        x = Dense(units_small, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)

        transformer_model = tf.keras.Model(inputs, outputs)
        transformer_model.compile(optimizer='adam', loss='mse')
        transformer_model.fit(X_train, y_train, epochs=epochs // 2, batch_size=batch_size, verbose=0, callbacks=callbacks, validation_split=validation_split)

        trans_pred_scaled = transformer_model.predict(X_test, verbose=0)
        trans_pred = scaler.inverse_transform(trans_pred_scaled).flatten()

        mae = mean_absolute_error(y_test_actual, trans_pred)
        rmse = np.sqrt(mean_squared_error(y_test_actual, trans_pred))
        mape = mean_absolute_percentage_error(y_test_actual, trans_pred) * 100

        results.append({
            'Model': 'Transformer',
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE (%)': round(mape, 2),
            'Type': 'Deep Learning'
        })
        trained_models['Transformer'] = {'model': transformer_model, 'scaler': scaler, 'seq_length': seq_length}
    except Exception as e:
        st.warning(f"Transformer failed: {str(e)}")

    return results, trained_models


def train_neuralprophet_model(train_data, test_data, date_col, target_col, custom_holidays=None):
    """Train NeuralProphet model"""
    try:
        import warnings
        import logging
        warnings.filterwarnings('ignore')
        logging.getLogger('neuralprophet').setLevel(logging.ERROR)
        logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

        from neuralprophet import NeuralProphet, set_log_level
        set_log_level("ERROR")

        # Aggregate by date (NeuralProphet requires unique dates)
        train_agg = train_data.groupby(date_col)[target_col].sum().reset_index()
        test_agg = test_data.groupby(date_col)[target_col].sum().reset_index()

        # Create fresh DataFrames with numpy arrays to avoid PyTorch .view() error
        # NeuralProphet 0.9+ requires clean numpy-backed data, not pandas Series
        train_dates = pd.to_datetime(train_agg[date_col]).values
        train_values = train_agg[target_col].values.astype(np.float64)
        test_dates = pd.to_datetime(test_agg[date_col]).values
        test_values = test_agg[target_col].values.astype(np.float64)

        np_train = pd.DataFrame({
            'ds': train_dates,
            'y': train_values
        })
        np_test = pd.DataFrame({
            'ds': test_dates,
            'y': test_values
        })

        # Sort and ensure contiguous memory layout
        np_train = np_train.sort_values('ds').reset_index(drop=True).copy()
        np_test = np_test.sort_values('ds').reset_index(drop=True).copy()

        model = NeuralProphet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            epochs=50,
            learning_rate=0.1,
            batch_size=min(32, len(np_train)),
            loss_func='MSE',
            normalize='standardize',
            trainer_config={'accelerator': 'cpu'},
        )

        # Add custom holidays if provided (avoids holidays library compatibility issue)
        if custom_holidays is not None and len(custom_holidays) > 0:
            # NeuralProphet expects 'event' column instead of 'holiday'
            holidays_df = custom_holidays.copy()
            if 'holiday' in holidays_df.columns:
                holidays_df = holidays_df.rename(columns={'holiday': 'event'})
            model = model.add_events(holidays_df['event'].unique().tolist())
            np_train = model.create_df_with_events(np_train, holidays_df)
            st.text(f"NeuralProphet: Added {len(custom_holidays)} custom holidays")

        model.fit(np_train, freq='D')

        forecast = model.predict(np_test)
        predictions = forecast['yhat1'].values.astype(np.float64)
        actuals = np_test['y'].values.astype(np.float64)

        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = mean_absolute_percentage_error(actuals, predictions) * 100

        return {
            'Model': 'NeuralProphet',
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE (%)': round(mape, 2),
            'Type': 'Deep Learning'
        }, model
    except Exception as e:
        error_msg = str(e)
        if 'holidays' in error_msg.lower():
            st.warning("NeuralProphet skipped: holidays library compatibility issue. Use Prophet instead for holiday support.")
            status_msg = '⚠️ Skipped: holidays library issue'
        else:
            st.warning(f"NeuralProphet failed: {error_msg}")
            status_msg = f'⚠️ Failed: {error_msg[:50]}'
        # Return skipped result for display in comparison table
        return {
            'Model': 'NeuralProphet',
            'MAE': '-',
            'RMSE': '-',
            'MAPE (%)': '-',
            'Type': 'Deep Learning',
            'Status': status_msg
        }, None


def train_time_series_models(train_data, test_data, date_col, target_col):
    """Train time series models"""
    results = []
    trained_models = {}

    # Aggregate by date first (handles duplicate dates from category combinations)
    train_agg = train_data.groupby(date_col)[target_col].sum().reset_index()
    test_agg = test_data.groupby(date_col)[target_col].sum().reset_index()

    # Create series with proper DatetimeIndex and frequency
    train_series = train_agg.set_index(date_col)[target_col].copy()
    test_series = test_agg.set_index(date_col)[target_col].copy()

    # Ensure index is DatetimeIndex and set frequency
    train_series.index = pd.DatetimeIndex(train_series.index)
    test_series.index = pd.DatetimeIndex(test_series.index)

    # Sort by date
    train_series = train_series.sort_index()
    test_series = test_series.sort_index()

    # Infer frequency from data (daily, weekly, etc.)
    try:
        inferred_freq = pd.infer_freq(train_series.index)
        if inferred_freq is None:
            inferred_freq = 'D'  # Default to daily
        train_series = train_series.asfreq(inferred_freq, method='ffill')
        test_series = test_series.asfreq(inferred_freq, method='ffill')
    except Exception:
        # If frequency inference fails, resample to daily
        train_series = train_series.resample('D').sum().ffill()
        test_series = test_series.resample('D').sum().ffill()

    # Auto ARIMA - automatically finds optimal (p,d,q) using ACF/PACF analysis
    try:
        from pmdarima import auto_arima

        st.text("Finding optimal ARIMA parameters using ACF/PACF analysis...")
        arima_model = auto_arima(
            train_series,
            start_p=0, max_p=5,
            start_q=0, max_q=5,
            d=None,  # Auto-detect differencing using ADF test
            seasonal=False,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            information_criterion='aic'
        )

        arima_order = arima_model.order
        arima_pred = arima_model.predict(n_periods=len(test_series))

        mae = mean_absolute_error(test_series, arima_pred)
        rmse = np.sqrt(mean_squared_error(test_series, arima_pred))
        mape = mean_absolute_percentage_error(test_series, arima_pred) * 100

        model_name = f'ARIMA{arima_order}'
        results.append({
            'Model': model_name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE (%)': round(mape, 2),
            'Type': 'Time Series'
        })
        trained_models['ARIMA'] = arima_model
        st.text(f"Optimal ARIMA parameters found: {arima_order} (p,d,q)")
    except Exception as e:
        st.warning(f"Auto ARIMA failed: {str(e)}")
        results.append({
            'Model': 'Auto ARIMA',
            'MAE': '-',
            'RMSE': '-',
            'MAPE (%)': '-',
            'Type': 'Time Series',
            'Status': f'⚠️ Failed: {str(e)[:50]}'
        })

    # Auto SARIMA - automatically finds optimal (p,d,q)(P,D,Q,m) using ACF/PACF analysis
    try:
        from pmdarima import auto_arima

        st.text("Finding optimal SARIMA parameters using ACF/PACF analysis...")
        sarima_model = auto_arima(
            train_series,
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            d=None,  # Auto-detect differencing
            seasonal=True,
            m=7,  # Weekly seasonality (can be made dynamic based on data)
            start_P=0, max_P=2,
            start_Q=0, max_Q=2,
            D=None,  # Auto-detect seasonal differencing
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            information_criterion='aic'
        )

        sarima_order = sarima_model.order
        sarima_seasonal = sarima_model.seasonal_order
        sarima_pred = sarima_model.predict(n_periods=len(test_series))

        mae = mean_absolute_error(test_series, sarima_pred)
        rmse = np.sqrt(mean_squared_error(test_series, sarima_pred))
        mape = mean_absolute_percentage_error(test_series, sarima_pred) * 100

        model_name = f'SARIMA{sarima_order}{sarima_seasonal}'
        results.append({
            'Model': model_name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE (%)': round(mape, 2),
            'Type': 'Time Series'
        })
        trained_models['SARIMA'] = sarima_model
        st.text(f"Optimal SARIMA parameters found: {sarima_order}x{sarima_seasonal}")
    except Exception as e:
        st.warning(f"Auto SARIMA failed: {str(e)}")
        results.append({
            'Model': 'Auto SARIMA',
            'MAE': '-',
            'RMSE': '-',
            'MAPE (%)': '-',
            'Type': 'Time Series',
            'Status': f'⚠️ Failed: {str(e)[:50]}'
        })

    # Exponential Smoothing (Holt-Winters)
    try:
        if len(train_series) >= 14:
            hw_model = ExponentialSmoothing(
                train_series,
                trend='add',
                seasonal='add',
                seasonal_periods=7
            )
            hw_fit = hw_model.fit()
            hw_pred = hw_fit.forecast(steps=len(test_series))

            mae = mean_absolute_error(test_series, hw_pred)
            rmse = np.sqrt(mean_squared_error(test_series, hw_pred))
            mape = mean_absolute_percentage_error(test_series, hw_pred) * 100

            results.append({
                'Model': 'Holt-Winters',
                'MAE': round(mae, 2),
                'RMSE': round(rmse, 2),
                'MAPE (%)': round(mape, 2),
                'Type': 'Time Series'
            })
            trained_models['Holt-Winters'] = hw_fit
    except Exception as e:
        st.warning(f"Holt-Winters failed: {str(e)}")
        results.append({
            'Model': 'Holt-Winters',
            'MAE': '-',
            'RMSE': '-',
            'MAPE (%)': '-',
            'Type': 'Time Series',
            'Status': f'⚠️ Failed: {str(e)[:50]}'
        })

    return results, trained_models, train_series, test_series


def train_prophet_model(train_data, test_data, date_col, target_col, holiday_country=None, custom_holidays=None):
    """Train Prophet model with optional holiday support"""
    try:
        from prophet import Prophet

        # Aggregate by date (Prophet requires unique dates)
        train_agg = train_data.groupby(date_col)[target_col].sum().reset_index()
        test_agg = test_data.groupby(date_col)[target_col].sum().reset_index()

        prophet_train = train_agg.rename(columns={date_col: 'ds', target_col: 'y'})
        prophet_test = test_agg.rename(columns={date_col: 'ds', target_col: 'y'})

        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

        # Add country holidays
        if holiday_country:
            model.add_country_holidays(country_name=holiday_country)
            st.text(f"Prophet: Added {holiday_country} holidays")

        # Add custom holidays
        if custom_holidays is not None and len(custom_holidays) > 0:
            model.add_holidays(custom_holidays)
            st.text(f"Prophet: Added {len(custom_holidays)} custom holidays")

        model.fit(prophet_train)

        forecast = model.predict(prophet_test[['ds']])
        predictions = forecast['yhat'].values

        mae = mean_absolute_error(prophet_test['y'], predictions)
        rmse = np.sqrt(mean_squared_error(prophet_test['y'], predictions))
        mape = mean_absolute_percentage_error(prophet_test['y'], predictions) * 100

        return {
            'Model': 'Prophet',
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE (%)': round(mape, 2),
            'Type': 'Time Series'
        }, model
    except Exception as e:
        st.warning(f"Prophet failed: {str(e)}")
        return {
            'Model': 'Prophet',
            'MAE': '-',
            'RMSE': '-',
            'MAPE (%)': '-',
            'Type': 'Time Series',
            'Status': f'⚠️ Failed: {str(e)[:50]}'
        }, None


def forecast_future(best_model, model_name, model_type, df, future_dates,
                    date_col, target_col, feature_cols, encoders, categorical_cols,
                    train_series=None):
    """Generate future forecasts using the best model"""

    if model_type == 'Machine Learning':
        future_df = pd.DataFrame({date_col: future_dates})
        future_df = preprocess_data(future_df, date_col, target_col)

        for col in categorical_cols:
            if col in df.columns:
                mode_result = df[col].mode()
                if len(mode_result) > 0:
                    future_df[col] = mode_result.iloc[0]
                else:
                    future_df[col] = df[col].iloc[0] if len(df[col]) > 0 else 'Unknown'
                if col in encoders:
                    # Handle unseen categories by using -1 as fallback
                    try:
                        future_df[col + '_encoded'] = encoders[col].transform(future_df[col].astype(str)).astype('float64')
                    except ValueError:
                        # Unseen category - use 0 (first known category encoding)
                        future_df[col + '_encoded'] = 0.0

        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in future_df.columns and col != target_col:
                col_mean = df[col].mean()
                future_df[col] = col_mean if pd.notna(col_mean) else 0.0

        available_features = [f for f in feature_cols if f in future_df.columns]
        X_future = future_df[available_features]
        predictions = best_model.predict(X_future)

    elif model_type == 'Deep Learning':
        if model_name == 'NeuralProphet':
            future_df = pd.DataFrame({'ds': future_dates})
            forecast = best_model.predict(future_df)
            predictions = forecast['yhat1'].values
        else:
            model_info = best_model
            model = model_info['model']
            scaler = model_info['scaler']
            seq_length = model_info['seq_length']
            is_flat = model_info.get('flat', False)

            last_sequence = scaler.transform(train_series.values[-seq_length:].reshape(-1, 1)).flatten()
            predictions = []

            for _ in range(len(future_dates)):
                if is_flat:
                    X_pred = last_sequence.reshape(1, -1)
                else:
                    X_pred = last_sequence.reshape(1, seq_length, 1)

                pred_scaled = model.predict(X_pred, verbose=0)[0, 0]
                pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
                predictions.append(pred)

                last_sequence = np.append(last_sequence[1:], pred_scaled)

            predictions = np.array(predictions)

    elif model_name == 'Prophet':
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = best_model.predict(future_df)
        predictions = forecast['yhat'].values

    else:  # Time series models
        predictions = best_model.forecast(steps=len(future_dates))

    return predictions


# Sidebar
st.sidebar.header("Configuration")

data_source = st.sidebar.radio(
    "Data Source",
    ["Upload CSV/Excel", "Use Sample Data"]
)

if data_source == "Upload CSV/Excel":
    uploaded_file = st.sidebar.file_uploader(
        "Upload your data",
        type=['csv', 'xlsx', 'xls']
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Validate uploaded data
            if df.empty:
                st.sidebar.error("Uploaded file is empty. Please upload a file with data.")
                df = None
            elif len(df) < 10:
                st.sidebar.warning(f"Only {len(df)} rows loaded. Minimum 10 rows recommended for forecasting.")

            if df is not None:
                # Convert ALL numeric columns to float64 to avoid int/float type issues
                for col in df.columns:
                    # Try to convert to numeric, keep as-is if it fails (text columns)
                    try:
                        if df[col].dtype in ['int64', 'int32', 'float64', 'float32'] or pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                    except Exception:
                        pass

                st.sidebar.success(f"Loaded {len(df)} rows")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            df = None
    else:
        df = None
else:
    df = generate_sample_data()
    st.sidebar.info("Using generated sample data")

# Model selection
st.sidebar.markdown("---")
st.sidebar.header("Model Selection")

# Show memory warning if in low memory mode
if LOW_MEMORY_MODE:
    st.sidebar.warning(f"⚠️ Low memory detected ({get_available_memory_gb():.1f}GB). Deep Learning disabled by default.")

use_ml = st.sidebar.checkbox("Machine Learning Models", value=True,
                              help="Best when you have multiple features (City, Channel, etc.) that influence the target")
use_ts = st.sidebar.checkbox("Time Series Models", value=True,
                              help="Best for data with clear trends and seasonal patterns")
use_dl = st.sidebar.checkbox("Deep Learning Models", value=not LOW_MEMORY_MODE,
                              help="Best for complex patterns and large datasets (500+ rows recommended). Requires 2GB+ RAM.",
                              disabled=False)

if use_dl and LOW_MEMORY_MODE:
    st.sidebar.error("⚠️ Enabling DL models in low memory may cause crashes. Consider using ML/Time Series models instead.")

# Advanced Settings
st.sidebar.markdown("---")
st.sidebar.header("Advanced Settings")

# Feature Engineering (Phase 1)
with st.sidebar.expander("🔧 Feature Engineering", expanded=True):
    use_lag_features = st.checkbox("Lag Features", value=True,
                                    help="Use yesterday's and last week's values to predict today")
    use_rolling_features = st.checkbox("Rolling Statistics", value=True,
                                        help="Use recent averages and trends to predict")
    use_outlier_detection = st.checkbox("Outlier Detection", value=True,
                                         help="Remove unusual spikes that could confuse the model")
    use_ensemble = st.checkbox("Model Ensemble", value=True,
                                help="Combine multiple models for more stable predictions")

    st.markdown("---")
    st.markdown("""
    **What These Mean (Simple):**

    📅 **Lag Features:** *"Yesterday was 500 calls, so today might be similar"*

    📈 **Rolling Statistics:** *"Last week averaged 480 calls, trend is going up"*

    🚫 **Outlier Detection:** *"Ignore that weird day with 10,000 calls - it was a glitch"*

    🤝 **Model Ensemble:** *"Ask 3 experts and average their opinions"*
    """)

# Production-grade options with detailed explanations
use_cross_validation = st.sidebar.checkbox("Cross-Validation (K-Fold)", value=False,
                                            help="Test model multiple times on different data portions for reliable accuracy")
use_hyperparameter_tuning = st.sidebar.checkbox("Hyperparameter Tuning (Optuna)", value=False,
                                                 help="Automatically find the best model settings for maximum accuracy")
use_prediction_intervals = st.sidebar.checkbox("Prediction Intervals (95% CI)", value=False,
                                                help="Show range of possible values (best case to worst case) for each forecast")

# Simple explanation for non-technical users
with st.sidebar.expander("❓ What Do These Options Mean?"):
    st.markdown("""
    ### 🔄 Cross-Validation (K-Fold)

    **Simple Explanation:**
    Instead of testing your model once, it tests **5 times** on different portions of your data.

    **Everyday Analogy:**
    *Like a student taking 5 practice tests instead of 1 to know their true skill level.*

    | When to Use | When to Skip |
    |-------------|--------------|
    | Small data (less than 500 rows) | Large data (5000+ rows) |
    | Need trustworthy results | Quick testing |
    | Final model selection | First exploration |

    ⏱️ **Time:** Takes 5x longer

    ---

    ### ⚙️ Hyperparameter Tuning (Optuna)

    **Simple Explanation:**
    Every model has "settings" (like volume on a TV). This feature **automatically finds the best settings** for your specific data.

    **Everyday Analogy:**
    *Like auto-tuning a guitar - it tries different adjustments until it sounds perfect.*

    | When to Use | When to Skip |
    |-------------|--------------|
    | Building final model | Quick testing |
    | Need best possible accuracy | First exploration |
    | Have time to wait | Time is limited |

    ⏱️ **Time:** Takes 10-20x longer (worth it for important forecasts!)

    ---

    ### 📊 Prediction Intervals (95% CI)

    **Simple Explanation:**
    Instead of just "tomorrow = 500 calls", you get "tomorrow = 450 to 550 calls (95% confident)".

    **Everyday Analogy:**
    *Like weather forecast saying "70-80°F" instead of just "75°F" - shows the uncertainty.*

    | When to Use | When to Skip |
    |-------------|--------------|
    | Planning staffing/budget | Just need rough estimate |
    | Risk-sensitive decisions | Quick overview |
    | Want to know uncertainty | Simple reporting |

    ---

    ### 🚀 Recommended Workflow

    **First Run:** All OFF → Quick results (30 seconds)

    **Second Run:** Cross-Validation ON → Verify accuracy (2 minutes)

    **Final Run:** Both ON → Best possible model (10-15 minutes)
    """)

# Phase 3: Explainability & Monitoring
with st.sidebar.expander("🔍 Explainability & Monitoring"):
    use_shap = st.checkbox("SHAP Explainability", value=False,
                           help="Understand WHY the model predicts what it does")
    use_drift_detection = st.checkbox("Data Drift Detection", value=False,
                                       help="Check if your data patterns have changed and model needs retraining")

    st.markdown("""
    ---
    ### 🔍 SHAP Explainability

    **Simple Explanation:**
    Shows **which factors** influenced each prediction the most.

    **Everyday Analogy:**
    *Like a doctor explaining "Your blood pressure is high because of salt intake and stress" - not just "it's high".*

    **Example Output:**
    - Day of week: +50 calls
    - Previous day volume: +30 calls
    - Holiday proximity: -20 calls
    - **Total Prediction: 500 calls**

    ---

    ### 🚨 Data Drift Detection

    **Simple Explanation:**
    Checks if your **recent data looks different** from what the model learned from.

    **Everyday Analogy:**
    *Like a GPS that warns "Map may be outdated" when roads have changed.*

    **When It Alerts You:**
    - ✅ Green = Data looks normal, model is reliable
    - ⚠️ Yellow = Some changes detected, monitor closely
    - 🔴 Red = Significant changes, consider retraining
    """)

# Holiday Configuration
with st.sidebar.expander("🎄 Holiday Configuration"):
    holiday_option = st.radio(
        "Holiday Settings",
        ["No holidays", "Select country", "Custom holidays"],
        help="Holidays improve forecasting for business data with holiday effects"
    )

    holiday_country = None
    custom_holidays = None

    if holiday_option == "Select country":
        holiday_country = st.selectbox(
            "Select Country",
            ["US", "IN", "UK", "DE", "FR", "CA", "AU", "JP", "CN", "BR", "MX"],
            help="US=USA, IN=India, UK=United Kingdom, DE=Germany, etc."
        )
        st.caption(f"Will use {holiday_country} public holidays automatically")

    elif holiday_option == "Custom holidays":
        st.markdown("Enter holiday dates (one per line, YYYY-MM-DD format):")
        holiday_text = st.text_area(
            "Custom Holiday Dates",
            placeholder="2024-12-25\n2024-01-01\n2024-11-28",
            height=100
        )
        if holiday_text.strip():
            try:
                custom_holidays = pd.DataFrame({
                    'ds': pd.to_datetime(holiday_text.strip().split('\n')),
                    'holiday': 'custom_holiday'
                })
                st.success(f"Loaded {len(custom_holidays)} custom holidays")
            except Exception as e:
                st.error(f"Invalid date format: {e}")

# Model Guide
with st.sidebar.expander("📖 Model Selection Guide"):
    st.markdown("### Machine Learning Models")
    st.markdown("""
    Use when you have **extra features** (City, Channel, AHT) that influence volume.

    | Model | Best For | Data Size |
    |-------|----------|-----------|
    | Linear Regression | Steady growth/decline trends | Any |
    | Ridge Regression | Many features, prevent overfitting | 100+ rows |
    | Random Forest | Non-linear patterns, feature importance | 200+ rows |
    | Gradient Boosting | Complex patterns, high accuracy | 300+ rows |
    | XGBoost | Production-grade predictions | 500+ rows |
    | LightGBM | Very large datasets, fast training | 1000+ rows |
    """)

    st.markdown("### Time Series Models")
    st.markdown("""
    Use when **date patterns** (trends, seasonality) are the main drivers.

    | Model | Best For | Example |
    |-------|----------|---------|
    | ARIMA | Data with trend, no clear seasonality | Sales with steady growth |
    | SARIMA | Clear weekly/monthly repeating patterns | Call volume (busy Mondays) |
    | Holt-Winters | Trend + seasonality combined | Retail sales with holiday spikes |
    | Prophet | Missing data, holidays, multiple seasons | Business metrics with weekends off |
    """)

    st.markdown("#### Understanding ARIMA(p,d,q)")
    st.markdown("""
    Parameters are **auto-detected** using ACF/PACF analysis:

    | Param | Name | Meaning | Found By |
    |-------|------|---------|----------|
    | **p** | AR order | Past values influencing today | PACF cutoff |
    | **d** | Differencing | Times to difference for stationarity | ADF test |
    | **q** | MA order | Past errors influencing today | ACF decay |

    *Example: ARIMA(2,1,1) = Last 2 days matter, differenced once, 1 error term*
    """)

    st.markdown("#### Understanding SARIMA(p,d,q)(P,D,Q,m)")
    st.markdown("""
    Adds **seasonal patterns** on top of ARIMA:

    | Param | Meaning |
    |-------|---------|
    | **(p,d,q)** | Short-term patterns (same as ARIMA) |
    | **(P,D,Q)** | Seasonal version at interval m |
    | **m** | Season length (7=weekly, 12=monthly) |

    *Example: SARIMA(1,0,1)(1,1,0,7) = Yesterday + same weekday last week matter, weekly cycle*
    """)

    st.markdown("### Deep Learning Models")
    st.markdown("""
    Use for **complex/long-term patterns** with sufficient data (500+ rows ideal).

    | Model | Best For | Example |
    |-------|----------|---------|
    | LSTM | Long memory needed (30+ day patterns) | Stock prices, weather |
    | GRU | Similar to LSTM but faster to train | Real-time forecasting |
    | 1D-CNN | Short-term local patterns matter most | Anomaly detection |
    | Dense NN | General purpose neural baseline | Any forecasting task |
    | Transformer | Different time periods have different importance | Demand with promotions |
    | NeuralProphet | Prophet's seasonality + neural network learning; handles trends, holidays, AND learns complex patterns automatically | Call centers with holidays + special events |
    """)

if df is not None:
    st.header("1. Data Preview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    col4.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))

    with st.expander("View Raw Data"):
        st.dataframe(df.head(100), use_container_width=True)

    st.header("2. Column Configuration")

    col1, col2 = st.columns(2)

    with col1:
        date_columns = [col for col in df.columns if 'date' in col.lower() or df[col].dtype == 'datetime64[ns]']
        if not date_columns:
            date_columns = df.columns.tolist()

        date_col = st.selectbox(
            "Select Date Column",
            options=date_columns,
            index=0
        )

    with col2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns found in the data. Please upload data with at least one numeric column for forecasting.")
            st.stop()
        target_col = st.selectbox(
            "Select Target Column (Volume to predict)",
            options=numeric_cols,
            index=0
        )

    categorical_cols = st.multiselect(
        "Select Categorical Columns (for forecast output)",
        options=[col for col in df.columns if col not in [date_col, target_col]],
        default=[col for col in ['Call_Type', 'City', 'Language', 'Channel'] if col in df.columns],
        help="These columns will be included in forecast output using most common values from historical data"
    )

    # Option to also forecast AHT or other numeric columns
    other_numeric_cols = [col for col in numeric_cols if col != target_col]
    additional_forecast_cols = []
    if other_numeric_cols:
        additional_forecast_cols = st.multiselect(
            "Also forecast these columns (optional)",
            options=other_numeric_cols,
            default=[],
            help="Select additional numeric columns to forecast (e.g., AHT)"
        )

    # Categories selected = include them in forecast output
    forecast_by_categories = categorical_cols

    st.header("3. Forecast Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100

    with col2:
        forecast_days = st.number_input("Forecast Days", min_value=1, max_value=365, value=30)

    with col3:
        df[date_col] = pd.to_datetime(df[date_col])
        min_date = df[date_col].max() + timedelta(days=1)
        max_date = min_date + timedelta(days=365)

        forecast_start = st.date_input(
            "Forecast Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )

    if st.button("🚀 Train Models & Generate Forecast", type="primary"):
        with st.spinner("Processing data and training models..."):

            # Step 1: Basic preprocessing
            df_processed = preprocess_data(df, date_col, target_col)

            # Validate target column
            if df_processed[target_col].isna().all():
                st.error(f"Target column '{target_col}' contains only missing values. Please check your data.")
                st.stop()
            nan_count = df_processed[target_col].isna().sum()
            if nan_count > 0:
                df_processed = df_processed.dropna(subset=[target_col])
                st.warning(f"Dropped {nan_count} rows with missing values in '{target_col}'")
            if (df_processed[target_col] == 0).all():
                st.error(f"Target column '{target_col}' contains only zeros. Cannot calculate MAPE.")
                st.stop()

            # Step 2: Outlier detection (before feature engineering)
            outlier_count = 0
            if use_outlier_detection:
                st.text("Detecting and handling outliers...")
                df_processed, outlier_count = detect_and_handle_outliers(df_processed, target_col)
                if outlier_count > 0:
                    st.text(f"Capped {outlier_count} outliers using IQR method")

            # Step 3: Feature Engineering - Lag features
            if use_lag_features:
                st.text("Adding lag features (1, 2, 3, 7, 14, 30 days)...")
                df_processed = add_lag_features(df_processed, target_col)

            # Step 4: Feature Engineering - Rolling statistics
            if use_rolling_features:
                st.text("Adding rolling statistics (7, 14, 30 day windows)...")
                df_processed = add_rolling_features(df_processed, target_col)

            # Step 5: Handle NaN values created by lag/rolling features
            if use_lag_features or use_rolling_features:
                initial_len = len(df_processed)
                df_processed = df_processed.dropna()
                dropped = initial_len - len(df_processed)
                if dropped > 0:
                    st.text(f"Dropped {dropped} rows with NaN values (from lag/rolling features)")

            # Step 6: Encode categorical
            df_processed, encoders = encode_categorical(df_processed, categorical_cols)

            train_size = int(len(df_processed) * (1 - test_size))
            train_data = df_processed.iloc[:train_size]
            test_data = df_processed.iloc[train_size:]

            # Validate train/test split
            if len(train_data) < 5:
                st.error(f"Not enough training data ({len(train_data)} rows). Need at least 5 rows. Try reducing test size or adding more data.")
                st.stop()
            if len(test_data) < 1:
                st.error(f"No test data available. Try reducing test size.")
                st.stop()

            all_results = []
            all_models = {}
            train_series = None
            feature_cols = []  # Initialize empty, will be populated if use_ml=True

            # Train ML models
            if use_ml:
                st.subheader("Training Machine Learning Models...")
                if use_cross_validation:
                    st.text("Using 5-Fold Time Series Cross-Validation...")
                if use_hyperparameter_tuning:
                    st.text("Optuna hyperparameter tuning enabled (20 trials per model)...")

                feature_cols = get_ml_features(df_processed, target_col, date_col, categorical_cols)

                X_train = train_data[feature_cols]
                y_train = train_data[target_col]
                X_test = test_data[feature_cols]
                y_test = test_data[target_col]

                ml_results, ml_models = train_ml_models(
                    X_train, y_train, X_test, y_test,
                    use_cv=use_cross_validation,
                    use_tuning=use_hyperparameter_tuning
                )
                all_results.extend(ml_results)
                all_models.update(ml_models)

            # Train Time Series models
            if use_ts:
                st.subheader("Training Time Series Models...")
                ts_results, ts_models, train_series, test_series = train_time_series_models(
                    train_data, test_data, date_col, target_col
                )
                all_results.extend(ts_results)
                all_models.update(ts_models)

                prophet_result, prophet_model = train_prophet_model(
                    train_data, test_data, date_col, target_col,
                    holiday_country=holiday_country,
                    custom_holidays=custom_holidays
                )
                if prophet_result:
                    all_results.append(prophet_result)
                    if prophet_model is not None:
                        all_models['Prophet'] = prophet_model

            # Train Deep Learning models
            if use_dl:
                st.subheader("Training Deep Learning Models...")

                if train_series is None:
                    train_series = train_data.set_index(date_col)[target_col]
                    test_series = test_data.set_index(date_col)[target_col]

                dl_results, dl_models = train_deep_learning_models(
                    train_series, test_series,
                    use_tuning=use_hyperparameter_tuning,
                    low_memory=LOW_MEMORY_MODE
                )
                all_results.extend(dl_results)
                all_models.update(dl_models)

                np_result, np_model = train_neuralprophet_model(
                    train_data, test_data, date_col, target_col,
                    custom_holidays=custom_holidays
                )
                if np_result:
                    all_results.append(np_result)
                    if np_model is not None:
                        all_models['NeuralProphet'] = np_model

            # Create Ensemble Model (weighted average of top 3 ML models)
            if use_ensemble and use_ml and len(all_results) >= 3:
                st.subheader("Creating Ensemble Model...")
                try:
                    ml_results_df = pd.DataFrame([r for r in all_results if r['Type'] == 'Machine Learning'])
                    if len(ml_results_df) >= 3:
                        ensemble_pred = create_ensemble_prediction(
                            all_models, X_test, ml_results_df, top_n=3
                        )
                        if ensemble_pred is not None:
                            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                            ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
                            ensemble_mape = mean_absolute_percentage_error(y_test, ensemble_pred) * 100

                            all_results.append({
                                'Model': 'Ensemble (Top 3)',
                                'MAE': round(ensemble_mae, 2),
                                'RMSE': round(ensemble_rmse, 2),
                                'MAPE (%)': round(ensemble_mape, 2),
                                'Type': 'Ensemble'
                            })
                            # Store ensemble info for forecasting
                            all_models['Ensemble (Top 3)'] = {
                                'models': all_models,
                                'top_models': ml_results_df.nsmallest(3, 'MAPE (%)')['Model'].tolist()
                            }
                            st.text(f"Ensemble created from top 3 models with MAPE: {ensemble_mape:.2f}%")
                except Exception as e:
                    st.warning(f"Ensemble creation failed: {str(e)}")

            if not all_results:
                st.error("No models were successfully trained. Please check your data.")
            else:
                st.header("4. Model Comparison Results")

                results_df = pd.DataFrame(all_results)

                # Separate trained models from skipped models
                trained_mask = results_df['MAPE (%)'].apply(lambda x: x != '-' and pd.notna(x))
                trained_df = results_df[trained_mask].copy()
                skipped_df = results_df[~trained_mask].copy()

                # Sort trained models by MAPE
                if len(trained_df) > 0:
                    trained_df['MAPE (%)'] = pd.to_numeric(trained_df['MAPE (%)'], errors='coerce')
                    trained_df = trained_df.sort_values('MAPE (%)')
                    trained_df['When to Use'] = trained_df['Model'].apply(get_model_description)
                    trained_df['Status'] = '✅ Trained'

                # Add description for skipped models
                if len(skipped_df) > 0:
                    skipped_df['When to Use'] = skipped_df['Model'].apply(get_model_description)

                # Combine: trained first, then skipped
                results_df = pd.concat([trained_df, skipped_df], ignore_index=True)

                # Reorder columns for better display
                display_cols = ['Model', 'Type', 'MAE', 'RMSE', 'MAPE (%)', 'Status']
                if 'CV MAE' in results_df.columns:
                    display_cols.insert(5, 'CV MAE')
                display_cols.append('When to Use')
                results_df = results_df[[c for c in display_cols if c in results_df.columns]]

                # Display with styling (only highlight numeric columns)
                if len(trained_df) > 0:
                    st.dataframe(results_df, use_container_width=True, height=400)
                else:
                    st.dataframe(results_df, use_container_width=True, height=400)

                # Show alert for skipped models
                if len(skipped_df) > 0:
                    skipped_models = skipped_df['Model'].tolist()
                    st.warning(f"⚠️ **{len(skipped_models)} model(s) skipped due to insufficient data:** {', '.join(skipped_models)}")

                # Get best model from trained models only
                if len(trained_df) > 0:
                    best_idx = trained_df['MAPE (%)'].idxmin()
                    best_row = trained_df.loc[best_idx]
                    best_model_name = best_row['Model']
                    best_model_type = best_row['Type']
                    best_mape = best_row['MAPE (%)']
                    best_mae = best_row['MAE']
                    best_rmse = best_row['RMSE']
                    best_description = best_row['When to Use']

                    st.success(f"🏆 Best Model: **{best_model_name}** with MAPE: {best_mape}%")
                    st.info(f"💡 **Why this model?** {best_description}")
                else:
                    st.error("No models were successfully trained. Please check your data or adjust settings.")
                    st.stop()

                st.header("5. Model Performance Visualization")

                # Only plot trained models (exclude skipped models with '-' MAPE)
                chart_df = trained_df.copy() if len(trained_df) > 0 else pd.DataFrame()
                if len(chart_df) > 0:
                    fig = px.bar(
                        chart_df,
                        x='Model',
                        y='MAPE (%)',
                        color='Type',
                        color_discrete_map={
                            'Machine Learning': '#636EFA',
                            'Time Series': '#EF553B',
                            'Deep Learning': '#00CC96',
                            'Ensemble': '#AB63FA'
                        },
                        title='Model Comparison by MAPE (%)'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No models were successfully trained to visualize.")


                st.header("6. Future Forecast")

                future_dates = pd.date_range(
                    start=forecast_start,
                    periods=forecast_days,
                    freq='D'
                )

                # Category-based forecasting
                if forecast_by_categories:
                    st.subheader("📊 Forecast Output (Same Format as Historical Data)")

                    import logging
                    logging.getLogger('prophet').setLevel(logging.WARNING)
                    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

                    # Step 1: Find ALL unique category combinations from historical data
                    st.text("Analyzing all category combinations from historical data...")
                    combo_group = df.groupby(forecast_by_categories)[target_col].sum().reset_index()
                    combo_group.columns = list(forecast_by_categories) + ['Historical_Volume']
                    total_historical_volume = combo_group['Historical_Volume'].sum()

                    # Check for zero volume
                    if total_historical_volume == 0 or pd.isna(total_historical_volume):
                        st.error(f"Total historical {target_col} is zero or missing. Cannot calculate proportions for category distribution.")
                        st.stop()

                    combo_group['Proportion'] = combo_group['Historical_Volume'] / total_historical_volume

                    num_combinations = len(combo_group)
                    st.success(f"Found {num_combinations} unique category combinations in historical data")

                    # Step 2: Predict TOTAL Volume using selected model (aggregate forecast)
                    st.text(f"Generating total {target_col} forecast using {best_model_name}...")
                    total_df = df.groupby(date_col)[target_col].sum().reset_index()
                    total_df = total_df.sort_values(date_col)

                    total_volume_predictions = None

                    try:
                        if best_model_name == 'Prophet' and 'Prophet' in all_models:
                            # Use trained Prophet model
                            future_df = pd.DataFrame({'ds': future_dates})
                            forecast = all_models['Prophet'].predict(future_df)
                            total_volume_predictions = forecast['yhat'].values

                        elif best_model_name == 'NeuralProphet' and 'NeuralProphet' in all_models:
                            # Use trained NeuralProphet model
                            future_df = pd.DataFrame({
                                'ds': pd.to_datetime(future_dates),
                                'y': np.zeros(len(future_dates))
                            })
                            forecast = all_models['NeuralProphet'].predict(future_df)
                            total_volume_predictions = forecast['yhat1'].values

                        elif best_model_type == 'Time Series' and best_model_name in all_models:
                            # ARIMA, SARIMA, Holt-Winters
                            model = all_models[best_model_name]
                            if hasattr(model, 'predict'):
                                total_volume_predictions = model.predict(n_periods=forecast_days)
                            elif hasattr(model, 'forecast'):
                                total_volume_predictions = model.forecast(steps=forecast_days)
                            total_volume_predictions = np.array(total_volume_predictions).flatten()

                        elif best_model_type == 'Machine Learning' and best_model_name in all_models:
                            # ML models need features - create future features
                            future_features = pd.DataFrame({'date': future_dates})
                            future_features['year'] = future_features['date'].dt.year.astype('float64')
                            future_features['month'] = future_features['date'].dt.month.astype('float64')
                            future_features['day'] = future_features['date'].dt.day.astype('float64')
                            future_features['dayofweek'] = future_features['date'].dt.dayofweek.astype('float64')
                            future_features['dayofyear'] = future_features['date'].dt.dayofyear.astype('float64')
                            future_features['weekofyear'] = future_features['date'].dt.isocalendar().week.astype('float64')
                            future_features['quarter'] = future_features['date'].dt.quarter.astype('float64')
                            future_features['is_weekend'] = (future_features['dayofweek'] >= 5).astype('float64')
                            future_features['is_month_start'] = future_features['date'].dt.is_month_start.astype('float64')
                            future_features['is_month_end'] = future_features['date'].dt.is_month_end.astype('float64')

                            # Match feature columns from training
                            ml_feature_cols = [c for c in feature_cols if c in future_features.columns]
                            missing_cols = [c for c in feature_cols if c not in future_features.columns]
                            for col in missing_cols:
                                future_features[col] = 0.0  # Fill missing with 0

                            total_volume_predictions = all_models[best_model_name].predict(future_features[feature_cols])

                        elif best_model_type == 'Deep Learning' and best_model_name in all_models:
                            # Deep Learning - use Prophet as fallback for aggregated forecast
                            from prophet import Prophet
                            prophet_df = total_df.rename(columns={date_col: 'ds', target_col: 'y'})
                            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                            model.fit(prophet_df)
                            future_df = pd.DataFrame({'ds': future_dates})
                            forecast = model.predict(future_df)
                            total_volume_predictions = forecast['yhat'].values
                            st.caption(f"Note: Using Prophet for aggregate forecast (Deep Learning models need special handling)")

                    except Exception as e:
                        st.warning(f"Error with {best_model_name}: {str(e)[:100]}. Using fallback...")

                    # Fallback to Prophet if selected model failed
                    if total_volume_predictions is None:
                        try:
                            from prophet import Prophet
                            prophet_df = total_df.rename(columns={date_col: 'ds', target_col: 'y'})
                            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                            model.fit(prophet_df)
                            future_df = pd.DataFrame({'ds': future_dates})
                            forecast = model.predict(future_df)
                            total_volume_predictions = forecast['yhat'].values
                        except Exception:
                            total_volume_predictions = np.full(len(future_dates), total_df[target_col].mean())

                    # Step 3: Predict additional columns if selected (e.g., AHT) - these are averages, not distributed
                    additional_predictions = {}
                    for add_col in additional_forecast_cols:
                        st.text(f"Generating {add_col} forecast...")
                        try:
                            add_df = df.groupby(date_col)[add_col].mean().reset_index()
                            add_df = add_df.sort_values(date_col)
                            prophet_add = add_df.rename(columns={date_col: 'ds', add_col: 'y'})
                            model_add = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                            model_add.fit(prophet_add)
                            add_forecast = model_add.predict(future_df)
                            additional_predictions[add_col] = add_forecast['yhat'].values
                        except Exception:
                            additional_predictions[add_col] = np.full(len(future_dates), df[add_col].mean())

                    # Step 4: Create forecast - ONE ROW PER COMBINATION PER DATE
                    # This is the CORRECT forecasting approach - distribute volume to each category combination
                    st.text(f"Distributing forecast across {num_combinations} category combinations...")

                    forecast_rows = []
                    for i, (date, daily_total) in enumerate(zip(future_dates, total_volume_predictions)):
                        for _, combo_row in combo_group.iterrows():
                            row = {'Date': date}
                            # Distribute volume based on historical proportion
                            row[target_col] = round(max(0, daily_total * combo_row['Proportion']), 2)
                            # Additional columns (like AHT) - same for all combinations on that day
                            for add_col in additional_forecast_cols:
                                row[add_col] = round(max(0, additional_predictions[add_col][i]), 2)
                            # Add category values for this combination
                            for cat_col in forecast_by_categories:
                                row[cat_col] = combo_row[cat_col]
                            forecast_rows.append(row)

                    forecast_output_df = pd.DataFrame(forecast_rows)

                    # Reorder columns: Date, Volume, AHT (if any), then categories
                    col_order = ['Date', target_col] + additional_forecast_cols + list(forecast_by_categories)
                    forecast_output_df = forecast_output_df[col_order]

                    # Display forecast
                    total_rows = len(forecast_output_df)
                    st.subheader("📋 Complete Forecast by Category Combination")
                    st.write(f"**{total_rows} rows** = {forecast_days} days × {num_combinations} combinations")

                    st.dataframe(forecast_output_df.head(200), use_container_width=True)
                    if total_rows > 200:
                        st.caption("Showing first 200 rows. Download CSV for complete data.")

                    # Create combined Historical + Forecast file
                    st.subheader("📥 Download Options")

                    # Prepare historical data (with category breakdown - same format as forecast)
                    historical_data = df[[date_col, target_col] + list(forecast_by_categories)].copy()
                    historical_data = historical_data.sort_values(date_col).reset_index(drop=True)
                    historical_data = historical_data.rename(columns={date_col: 'Date', target_col: 'Actual'})
                    historical_data['Forecast'] = ''

                    # Prepare forecast data
                    forecast_download = forecast_output_df.copy()
                    forecast_download = forecast_download.rename(columns={target_col: 'Forecast'})
                    forecast_download['Actual'] = ''

                    # Reorder columns to match: Date, Actual, Forecast, categories
                    col_order_combined = ['Date', 'Actual', 'Forecast'] + list(forecast_by_categories)
                    historical_data = historical_data[col_order_combined]
                    forecast_download = forecast_download[col_order_combined]

                    # Combine historical + forecast
                    combined_output = pd.concat([historical_data, forecast_download], ignore_index=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        # Combined file (Historical + Forecast)
                        csv_combined = combined_output.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Historical + Forecast Combined",
                            data=csv_combined,
                            file_name=f"historical_and_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            key="download_combined"
                        )
                        st.caption("Historical (Date, Actual) + Forecast (Date, Forecast, Categories)")

                    with col2:
                        # Forecast only
                        csv_forecast = forecast_output_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Forecast Only",
                            data=csv_forecast,
                            file_name=f"forecast_by_category_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            key="download_forecast_only"
                        )
                        st.caption("Forecast with category breakdown only")

                    # Step 5: Show summary by each category dimension
                    st.subheader("📊 Forecast Summary by Category")

                    tabs = st.tabs(forecast_by_categories)
                    for tab, cat_col in zip(tabs, forecast_by_categories):
                        with tab:
                            # Aggregate forecast by this category
                            cat_summary = forecast_output_df.groupby(cat_col)[target_col].agg(['sum', 'mean']).reset_index()
                            cat_summary.columns = [cat_col, f'Total_{target_col}', f'Avg_Daily_{target_col}']
                            cat_summary[f'Total_{target_col}'] = cat_summary[f'Total_{target_col}'].round(0)
                            cat_summary[f'Avg_Daily_{target_col}'] = cat_summary[f'Avg_Daily_{target_col}'].round(2)
                            total_sum = cat_summary[f'Total_{target_col}'].sum()
                            if total_sum > 0:
                                cat_summary['% of Total'] = (cat_summary[f'Total_{target_col}'] / total_sum * 100).round(1).astype(str) + '%'
                            else:
                                cat_summary['% of Total'] = '0%'

                            st.dataframe(cat_summary, use_container_width=True)

                            # Chart
                            fig = px.bar(
                                cat_summary,
                                x=cat_col,
                                y=f'Total_{target_col}',
                                title=f'Total Forecasted {target_col} by {cat_col}',
                                text=f'Total_{target_col}'
                            )
                            fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                            st.plotly_chart(fig, use_container_width=True)

                    # Step 6: Show all combinations with their proportions
                    st.subheader("📈 Category Combinations Distribution")
                    combo_display = combo_group.copy()
                    combo_display['Proportion'] = (combo_display['Proportion'] * 100).round(2).astype(str) + '%'
                    combo_display['Forecasted_Daily_Avg'] = (combo_group['Proportion'] * np.mean(total_volume_predictions)).round(2)
                    combo_display['Forecasted_Total'] = (combo_group['Proportion'] * sum(total_volume_predictions)).round(0)
                    st.dataframe(combo_display, use_container_width=True)

                    # Overall summary metrics
                    st.subheader("📊 Overall Forecast Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Forecast Days", forecast_days)
                    with col2:
                        st.metric("Category Combinations", num_combinations)
                    with col3:
                        st.metric(f"Avg Daily Total {target_col}", f"{np.mean(total_volume_predictions):.0f}")
                    with col4:
                        st.metric(f"Grand Total {target_col}", f"{sum(total_volume_predictions):.0f}")

                    if additional_predictions:
                        cols = st.columns(len(additional_predictions))
                        for i, (add_col, preds) in enumerate(additional_predictions.items()):
                            with cols[i]:
                                st.metric(f"Avg {add_col}", f"{np.mean(preds):.1f}")

                    # Download forecasts from ALL models
                    st.subheader("📥 Download Forecasts by Model")
                    st.write("Download forecast from each trained model (same format: Historical + Forecast with categories)")

                    # Prepare historical series for DL models
                    historical_series = df.groupby(date_col)[target_col].sum().sort_index()

                    # Generate forecasts for each model
                    all_model_forecasts = {}

                    for model_name, model in all_models.items():
                        try:
                            model_type = None
                            for r in all_results:
                                if r['Model'] == model_name:
                                    model_type = r['Type']
                                    break

                            model_total_predictions = None

                            if model_name == 'Prophet':
                                future_df_model = pd.DataFrame({'ds': future_dates})
                                forecast = model.predict(future_df_model)
                                model_total_predictions = forecast['yhat'].values

                            elif model_name == 'NeuralProphet':
                                np_future = pd.DataFrame({
                                    'ds': pd.to_datetime(future_dates),
                                    'y': np.zeros(len(future_dates))
                                })
                                forecast = model.predict(np_future)
                                model_total_predictions = forecast['yhat1'].values

                            elif model_name == 'Holt-Winters':
                                # Holt-Winters uses forecast() method
                                model_total_predictions = model.forecast(steps=forecast_days)
                                model_total_predictions = np.array(model_total_predictions).flatten()

                            elif model_name in ['ARIMA', 'Auto ARIMA']:
                                # pmdarima ARIMA uses predict()
                                model_total_predictions = model.predict(n_periods=forecast_days)
                                model_total_predictions = np.array(model_total_predictions).flatten()

                            elif model_name in ['SARIMA', 'Auto SARIMA']:
                                # pmdarima SARIMA uses predict()
                                model_total_predictions = model.predict(n_periods=forecast_days)
                                model_total_predictions = np.array(model_total_predictions).flatten()

                            elif model_type == 'Time Series':
                                # Generic time series fallback
                                if hasattr(model, 'forecast'):
                                    model_total_predictions = model.forecast(steps=forecast_days)
                                elif hasattr(model, 'predict'):
                                    try:
                                        model_total_predictions = model.predict(n_periods=forecast_days)
                                    except TypeError:
                                        model_total_predictions = model.predict(forecast_days)
                                if model_total_predictions is not None:
                                    model_total_predictions = np.array(model_total_predictions).flatten()

                            elif model_type == 'Deep Learning':
                                # Deep Learning models (LSTM, GRU, Transformer, Dense NN, etc.)
                                if isinstance(model, dict) and 'model' in model:
                                    dl_model = model['model']
                                    dl_scaler = model['scaler']
                                    seq_length = model['seq_length']
                                    is_flat = model.get('flat', False)  # Dense NN uses flat input

                                    # Scale historical data
                                    scaled_data = dl_scaler.transform(historical_series.values.reshape(-1, 1)).flatten()
                                    last_seq = scaled_data[-seq_length:]

                                    # Generate predictions iteratively
                                    predictions = []
                                    current_seq = last_seq.copy()
                                    for _ in range(forecast_days):
                                        if is_flat:
                                            # Dense NN: input shape (1, seq_length)
                                            pred = dl_model.predict(current_seq.reshape(1, seq_length), verbose=0)[0, 0]
                                        else:
                                            # LSTM, GRU, etc.: input shape (1, seq_length, 1)
                                            pred = dl_model.predict(current_seq.reshape(1, seq_length, 1), verbose=0)[0, 0]
                                        predictions.append(pred)
                                        current_seq = np.append(current_seq[1:], pred)

                                    # Inverse transform
                                    model_total_predictions = dl_scaler.inverse_transform(
                                        np.array(predictions).reshape(-1, 1)
                                    ).flatten()

                            elif model_name == 'Ensemble (Top 3)':
                                # Ensemble - use the stored predictions or skip
                                if isinstance(model, dict) and 'top_models' in model:
                                    # Generate predictions from each top model and average
                                    ensemble_preds = []
                                    for top_model_name in model['top_models']:
                                        if top_model_name in all_models:
                                            top_model = all_models[top_model_name]
                                            future_features = pd.DataFrame({'date': future_dates})
                                            future_features['year'] = future_features['date'].dt.year.astype('float64')
                                            future_features['month'] = future_features['date'].dt.month.astype('float64')
                                            future_features['day'] = future_features['date'].dt.day.astype('float64')
                                            future_features['dayofweek'] = future_features['date'].dt.dayofweek.astype('float64')
                                            future_features['dayofyear'] = future_features['date'].dt.dayofyear.astype('float64')
                                            future_features['weekofyear'] = future_features['date'].dt.isocalendar().week.astype('float64')
                                            future_features['quarter'] = future_features['date'].dt.quarter.astype('float64')
                                            future_features['is_weekend'] = (future_features['dayofweek'] >= 5).astype('float64')
                                            future_features['is_month_start'] = future_features['date'].dt.is_month_start.astype('float64')
                                            future_features['is_month_end'] = future_features['date'].dt.is_month_end.astype('float64')
                                            for col in feature_cols:
                                                if col not in future_features.columns:
                                                    future_features[col] = 0.0
                                            ensemble_preds.append(top_model.predict(future_features[feature_cols]))
                                    if ensemble_preds:
                                        model_total_predictions = np.mean(ensemble_preds, axis=0)

                            elif model_type == 'Machine Learning':
                                future_features = pd.DataFrame({'date': future_dates})
                                future_features['year'] = future_features['date'].dt.year.astype('float64')
                                future_features['month'] = future_features['date'].dt.month.astype('float64')
                                future_features['day'] = future_features['date'].dt.day.astype('float64')
                                future_features['dayofweek'] = future_features['date'].dt.dayofweek.astype('float64')
                                future_features['dayofyear'] = future_features['date'].dt.dayofyear.astype('float64')
                                future_features['weekofyear'] = future_features['date'].dt.isocalendar().week.astype('float64')
                                future_features['quarter'] = future_features['date'].dt.quarter.astype('float64')
                                future_features['is_weekend'] = (future_features['dayofweek'] >= 5).astype('float64')
                                future_features['is_month_start'] = future_features['date'].dt.is_month_start.astype('float64')
                                future_features['is_month_end'] = future_features['date'].dt.is_month_end.astype('float64')
                                for col in feature_cols:
                                    if col not in future_features.columns:
                                        future_features[col] = 0.0
                                model_total_predictions = model.predict(future_features[feature_cols])

                            if model_total_predictions is not None and len(model_total_predictions) == forecast_days:
                                # Create forecast rows with category distribution
                                model_forecast_rows = []
                                for i, (date, daily_total) in enumerate(zip(future_dates, model_total_predictions)):
                                    for _, combo_row in combo_group.iterrows():
                                        row = {'Date': date}
                                        row['Forecast'] = round(max(0, daily_total * combo_row['Proportion']), 2)
                                        for cat_col in forecast_by_categories:
                                            row[cat_col] = combo_row[cat_col]
                                        model_forecast_rows.append(row)

                                model_forecast_df = pd.DataFrame(model_forecast_rows)

                                # Combine with historical - keep category breakdown
                                historical_for_model = df[[date_col, target_col] + list(forecast_by_categories)].copy()
                                historical_for_model = historical_for_model.rename(columns={date_col: 'Date', target_col: 'Actual'})
                                historical_for_model['Forecast'] = ''

                                model_forecast_df['Actual'] = ''
                                col_order_model = ['Date', 'Actual', 'Forecast'] + list(forecast_by_categories)
                                historical_for_model = historical_for_model[col_order_model]
                                model_forecast_df = model_forecast_df[col_order_model]

                                # Sort historical by date
                                historical_for_model = historical_for_model.sort_values('Date').reset_index(drop=True)

                                combined_model_output = pd.concat([historical_for_model, model_forecast_df], ignore_index=True)
                                all_model_forecasts[model_name] = combined_model_output

                        except Exception as e:
                            st.caption(f"Could not generate forecast for {model_name}: {str(e)[:50]}")
                            continue

                    if all_model_forecasts:
                        # Create tabs for each model
                        model_tabs = st.tabs(list(all_model_forecasts.keys()))
                        for tab, (model_name, model_df) in zip(model_tabs, all_model_forecasts.items()):
                            with tab:
                                st.write(f"**{model_name}** - {len(model_df)} rows (Historical + Forecast)")
                                st.dataframe(model_df.head(100), use_container_width=True)

                                csv = model_df.to_csv(index=False)
                                # Sanitize model name for key and filename (remove special chars)
                                import re
                                safe_model_name = re.sub(r'[^a-zA-Z0-9_]', '_', model_name)
                                st.download_button(
                                    label=f"📥 Download {model_name} Forecast",
                                    data=csv,
                                    file_name=f"forecast_{safe_model_name.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv",
                                    key=f"download_model_{safe_model_name}"
                                )

                else:
                    # Original single forecast logic - use selected model
                    if best_model_name.startswith('ARIMA'):
                        best_model = all_models.get('ARIMA')
                    elif best_model_name.startswith('SARIMA'):
                        best_model = all_models.get('SARIMA')
                    else:
                        best_model = all_models.get(best_model_name)

                    if best_model is not None:
                        if train_series is None:
                            train_series = train_data.set_index(date_col)[target_col]

                        feature_cols = get_ml_features(df_processed, target_col, date_col, categorical_cols) if use_ml else []

                        predictions = forecast_future(
                            best_model, best_model_name, best_model_type,
                            df_processed, future_dates, date_col, target_col,
                            feature_cols, encoders, categorical_cols,
                            train_series=train_series
                        )

                        # Calculate prediction intervals if enabled and model is DL
                        lower_bound = None
                        upper_bound = None
                        if use_prediction_intervals and best_model_type == 'Deep Learning' and best_model_name not in ['NeuralProphet']:
                            try:
                                st.text("Calculating 95% prediction intervals (Monte Carlo Dropout)...")
                                model_info = best_model
                                if isinstance(model_info, dict) and 'model' in model_info:
                                    dl_model = model_info['model']
                                    dl_scaler = model_info['scaler']
                                    seq_length = model_info['seq_length']

                                    last_seq = dl_scaler.transform(train_series.values[-seq_length:].reshape(-1, 1)).flatten()
                                    future_X = []
                                    temp_seq = last_seq.copy()

                                    for _ in range(forecast_days):
                                        future_X.append(temp_seq.copy())
                                        next_pred = dl_model.predict(temp_seq.reshape(1, seq_length, 1), verbose=0)[0, 0]
                                        temp_seq = np.append(temp_seq[1:], next_pred)

                                    future_X = np.array(future_X).reshape(-1, seq_length, 1)
                                    predictions, lower_bound, upper_bound = calculate_prediction_intervals(
                                        dl_model, future_X, dl_scaler, n_iterations=50
                                    )
                            except Exception as e:
                                st.warning(f"Prediction intervals failed: {str(e)}")

                        forecast_df = pd.DataFrame({
                            'Date': future_dates,
                            'Predicted_Volume': predictions
                        })

                        if lower_bound is not None:
                            forecast_df['Lower_95%'] = lower_bound
                            forecast_df['Upper_95%'] = upper_bound

                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=df[date_col],
                            y=df[target_col],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue')
                        ))

                        if lower_bound is not None:
                            fig.add_trace(go.Scatter(
                                x=list(forecast_df['Date']) + list(forecast_df['Date'][::-1]),
                                y=list(upper_bound) + list(lower_bound[::-1]),
                                fill='toself',
                                fillcolor='rgba(255, 0, 0, 0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                hoverinfo='skip',
                                name='95% Confidence Interval'
                            ))

                        fig.add_trace(go.Scatter(
                            x=forecast_df['Date'],
                            y=forecast_df['Predicted_Volume'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red', dash='dash')
                        ))

                        fig.update_layout(
                            title=f'Historical Data & {forecast_days}-Day Forecast using {best_model_name}',
                            xaxis_title='Date',
                            yaxis_title=target_col,
                            hovermode='x unified'
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("Forecast Data")
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.dataframe(forecast_df, use_container_width=True)

                        with col2:
                            st.metric("Forecast Period", f"{forecast_days} days")
                            st.metric("Avg Predicted Volume", f"{predictions.mean():.0f}")
                            st.metric("Min Predicted", f"{predictions.min():.0f}")
                            st.metric("Max Predicted", f"{predictions.max():.0f}")
                            if lower_bound is not None:
                                avg_interval = np.mean(upper_bound - lower_bound)
                                st.metric("Avg 95% CI Width", f"±{avg_interval/2:.0f}")

                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Forecast CSV",
                            data=csv,
                            file_name=f"forecast_{best_model_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )

                    st.header("7. Summary")

                    summary_col1, summary_col2 = st.columns(2)

                    with summary_col1:
                        st.subheader("Model Details")
                        st.write(f"**Best Model:** {best_model_name}")
                        st.write(f"**Model Type:** {best_model_type}")
                        st.write(f"**MAPE:** {best_mape}%")
                        st.write(f"**MAE:** {best_mae}")
                        st.write(f"**RMSE:** {best_rmse}")

                    with summary_col2:
                        st.subheader("Data Summary")
                        st.write(f"**Training Samples:** {len(train_data)}")
                        st.write(f"**Test Samples:** {len(test_data)}")
                        st.write(f"**Total Models Trained:** {len(all_results)}")
                        st.write(f"**Forecast Period:** {forecast_days} days")

                    # Phase 3: SHAP Explainability
                    if use_shap and best_model_type == 'Machine Learning':
                        st.header("8. Model Explainability (SHAP)")
                        try:
                            st.text(f"Calculating SHAP values for {best_model_name}...")
                            shap_values, feature_importance, explainer = calculate_shap_values(
                                all_models[best_model_name], X_train, X_test, feature_cols, best_model_name
                            )

                            if feature_importance is not None:
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.subheader("Feature Importance")
                                    fig_importance = px.bar(
                                        feature_importance.head(15),
                                        x='Importance',
                                        y='Feature',
                                        orientation='h',
                                        title='Top 15 Features by SHAP Importance'
                                    )
                                    fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                                    st.plotly_chart(fig_importance, use_container_width=True)

                                with col2:
                                    st.subheader("How to Read SHAP")
                                    st.markdown("""
                                    **SHAP (SHapley Additive exPlanations)** shows:
                                    - **Higher bar** = Feature has more impact on predictions
                                    - **Positive SHAP** = Increases predicted volume
                                    - **Negative SHAP** = Decreases predicted volume

                                    **Example interpretation:**
                                    If `lag_1` has highest importance, yesterday's volume strongly predicts today's.
                                    """)

                                    # Show top 5 features explanation
                                    st.markdown("**Top 5 Most Important Features:**")
                                    for i, row in feature_importance.head(5).iterrows():
                                        st.write(f"• **{row['Feature']}**: Impact score = {row['Importance']:.3f}")
                            else:
                                st.warning("SHAP calculation not available for this model type")
                        except Exception as e:
                            st.warning(f"SHAP analysis failed: {str(e)}")

                    # Phase 3: Data Drift Detection
                    if use_drift_detection:
                        st.header("9. Data Drift Monitoring")
                        try:
                            st.text("Analyzing data distribution changes...")
                            drift_report, has_drift = detect_data_drift(train_data, test_data, target_col)

                            if has_drift:
                                st.warning("⚠️ **Data Drift Detected!** Some features have significantly different distributions in test data. Consider retraining the model.")
                            else:
                                st.success("✅ **No Significant Drift Detected.** Data distributions are stable.")

                            st.subheader("Drift Report")
                            st.dataframe(drift_report, use_container_width=True)

                            st.markdown("""
                            **Understanding the Report:**
                            - **KS Test**: Kolmogorov-Smirnov test compares distributions
                            - **P-Value < 0.05**: Significant difference detected (drift)
                            - **Action**: If drift detected, retrain model with recent data
                            """)

                            # Model health metrics (only if ML models were trained)
                            if use_ml and best_model_type == 'Machine Learning':
                                st.subheader("Model Health")
                                # Look up model - handle ARIMA/SARIMA dynamic names
                                model_key = best_model_name
                                if best_model_name.startswith('ARIMA') and 'SARIMA' not in best_model_name:
                                    model_key = 'ARIMA'
                                elif best_model_name.startswith('SARIMA'):
                                    model_key = 'SARIMA'

                                if model_key in all_models:
                                    health_metrics = calculate_model_monitoring_metrics(
                                        y_test, all_models[model_key].predict(X_test),
                                        baseline_mape=best_mape
                                    )
                                    health_col1, health_col2, health_col3 = st.columns(3)
                                    health_col1.metric("Current MAPE", f"{health_metrics['Current MAPE (%)']}%")
                                    health_col2.metric("Current MAE", f"{health_metrics['Current MAE']}")
                                    if 'Status' in health_metrics:
                                        health_col3.metric("Model Status", health_metrics['Status'])
                                else:
                                    st.info(f"Model health metrics not available for {best_model_name}")
                            else:
                                st.info("Model health metrics require Machine Learning models to be enabled.")

                        except Exception as e:
                            st.warning(f"Drift detection failed: {str(e)}")

else:
    st.info("👈 Please upload a CSV/Excel file or select 'Use Sample Data' to get started")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This forecasting pipeline supports:
- **ML Models**: Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Time Series**: ARIMA, SARIMA, Holt-Winters, Prophet
- **Deep Learning**: LSTM, GRU, 1D-CNN, Dense NN, Transformer, NeuralProphet
- **Metrics**: MAE, RMSE, MAPE
""")
