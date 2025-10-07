from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
import os
import time
import requests
warnings.filterwarnings('ignore')

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Configure requests session with proper headers
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
})

# Override yfinance's session
yf.utils.session = session

def convert_to_native(obj):
    """Convert numpy/pandas types to native Python types"""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj

class StockPredictor:
    def __init__(self, ticker_symbol="^GSPC"):
        self.ridge_model = None
        self.lstm_model = None
        self.scaler = None
        self.scaler_x = None
        self.scaler_y = None
        self.ticker_symbol = ticker_symbol.upper()

    def load_data(self):
        """Load and prepare stock data with enhanced session handling"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # Use 1 year instead of 300 days

            print(f"Loading data for {self.ticker_symbol} from {start_date.date()} to {end_date.date()}")

            # Try multiple methods to get data
            data = None
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    print(f"Attempt {attempt + 1}/{max_retries}")

                    # Method 1: Using download with session
                    ticker_obj = yf.Ticker(self.ticker_symbol, session=session)
                    data = ticker_obj.history(
                        period='1y',
                        interval='1d',
                        auto_adjust=True,
                        prepost=False
                    )

                    if not data.empty and len(data) > 30:
                        print(f"Success! Downloaded {len(data)} days of data")
                        break

                    # Wait before retry
                    time.sleep(2 * (attempt + 1))

                except Exception as e:
                    print(f"Attempt {attempt + 1} error: {str(e)[:100]}")
                    if attempt < max_retries - 1:
                        time.sleep(3)
                    continue

            if data is None or data.empty or len(data) < 30:
                print(f"Failed to load data for {self.ticker_symbol}")
                return False

            # Use Close price
            stock_close = data['Close'].dropna()

            if len(stock_close) < 30:
                print(f"Insufficient data: only {len(stock_close)} days")
                return False

            print(f"Using {len(stock_close)} days of data for training")

            # Create features for Ridge model
            features_df = pd.DataFrame({
                'stock_sma5': stock_close.rolling(5).mean().shift(1),
                'stock_sma20': stock_close.rolling(20).mean().shift(1)
            }).dropna()

            target = stock_close.loc[features_df.index]

            # Scale features for Ridge
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(features_df)
            y = target.values

            # Train Ridge model
            self.ridge_model = Ridge(solver='svd', alpha=1, random_state=69)
            self.ridge_model.fit(X_scaled, y)

            # Prepare LSTM data
            lookback = 20
            X_lstm, y_lstm = [], []

            for i in range(lookback, len(stock_close)):
                X_lstm.append(stock_close.iloc[i-lookback:i].values)
                y_lstm.append(stock_close.iloc[i])

            X_lstm = np.array(X_lstm)
            y_lstm = np.array(y_lstm)
            X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

            # Scale LSTM data
            self.scaler_x = MinMaxScaler()
            self.scaler_y = MinMaxScaler()

            num_samples, timesteps, num_features = X_lstm.shape
            X_flat = X_lstm.reshape(-1, num_features)
            X_scaled_flat = self.scaler_x.fit_transform(X_flat)
            X_scaled = X_scaled_flat.reshape(num_samples, timesteps, num_features)

            y_scaled = self.scaler_y.fit_transform(y_lstm.reshape(-1, 1))

            # Build and train LSTM model
            self.lstm_model = Sequential([
                LSTM(50, activation='tanh', input_shape=(timesteps, num_features)),
                Dense(1)
            ])

            self.lstm_model.compile(optimizer='adam', loss='mse')

            train_size = int(0.8 * len(X_scaled))
            self.lstm_model.fit(X_scaled[:train_size], y_scaled[:train_size], 
                              epochs=50, batch_size=16, verbose=0)

            self.stock_close = stock_close
            self.features_df = features_df
            print("Model training complete!")
            return True

        except Exception as e:
            print(f"Error in load_data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict_next_day(self):
        """Predict next trading day price"""
        try:
            if not all([self.ridge_model, self.lstm_model, self.scaler, 
                       self.scaler_x, self.scaler_y]):
                return None

            last_date = self.stock_close.index[-1]
            if isinstance(last_date, pd.Timestamp):
                last_date = last_date.to_pydatetime()

            current_price = self.stock_close.iloc[-1]

            next_date = last_date + timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)

            stock_sma5 = self.stock_close.rolling(5).mean().iloc[-1]
            stock_sma20 = self.stock_close.rolling(20).mean().iloc[-1]

            feature_row = pd.DataFrame([[stock_sma5, stock_sma20]], 
                                     columns=self.features_df.columns)

            feature_scaled = self.scaler.transform(feature_row)
            ridge_prediction = self.ridge_model.predict(feature_scaled)[0]

            last_sequence = self.stock_close.iloc[-20:].values.reshape(1, 20, 1)
            last_sequence_scaled = self.scaler_x.transform(
                last_sequence.reshape(-1, 1)).reshape(1, 20, 1)

            lstm_pred_scaled = self.lstm_model.predict(last_sequence_scaled, verbose=0)
            lstm_prediction = self.scaler_y.inverse_transform(lstm_pred_scaled)[0, 0]

            ridge_change = ridge_prediction - current_price
            lstm_change = lstm_prediction - current_price
            ridge_change_pct = (ridge_change / current_price) * 100
            lstm_change_pct = (lstm_change / current_price) * 100

            avg_prediction = (ridge_prediction + lstm_prediction) / 2
            avg_change = avg_prediction - current_price
            avg_change_pct = (avg_change / current_price) * 100

            ridge_positive = bool(ridge_change > 0)
            lstm_positive = bool(lstm_change > 0)
            models_agree = ridge_positive == lstm_positive

            result = {
                'ticker': self.ticker_symbol,
                'next_date': next_date.strftime('%Y-%m-%d'),
                'current_date': last_date.strftime('%Y-%m-%d'),
                'current_price': current_price,
                'ridge_prediction': ridge_prediction,
                'lstm_prediction': lstm_prediction,
                'average_prediction': avg_prediction,
                'ridge_change': ridge_change,
                'lstm_change': lstm_change,
                'average_change': avg_change,
                'ridge_change_pct': ridge_change_pct,
                'lstm_change_pct': lstm_change_pct,
                'average_change_pct': avg_change_pct,
                'ridge_direction': 'BULLISH' if ridge_positive else 'BEARISH',
                'lstm_direction': 'BULLISH' if lstm_positive else 'BEARISH',
                'models_agree': models_agree
            }

            result = convert_to_native(result)

            result['current_price'] = round(result['current_price'], 2)
            result['ridge_prediction'] = round(result['ridge_prediction'], 2)
            result['lstm_prediction'] = round(result['lstm_prediction'], 2)
            result['average_prediction'] = round(result['average_prediction'], 2)
            result['ridge_change'] = round(result['ridge_change'], 2)
            result['lstm_change'] = round(result['lstm_change'], 2)
            result['average_change'] = round(result['average_change'], 2)
            result['ridge_change_pct'] = round(result['ridge_change_pct'], 2)
            result['lstm_change_pct'] = round(result['lstm_change_pct'], 2)
            result['average_change_pct'] = round(result['average_change_pct'], 2)

            return result

        except Exception as e:
            print(f"Error making prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """API endpoint to get next day prediction"""
    try:
        if request.method == 'POST':
            data = request.get_json()
            ticker = data.get('ticker', '^GSPC')
        else:
            ticker = request.args.get('ticker', '^GSPC')

        ticker = ticker.strip().upper()
        if not ticker:
            ticker = '^GSPC'

        print(f"\n{'='*60}")
        print(f"Predicting for ticker: {ticker}")
        print('='*60)

        predictor = StockPredictor(ticker)

        if not predictor.load_data():
            return jsonify({
                'success': False, 
                'error': f'Unable to fetch data for {ticker}. Yahoo Finance may be temporarily unavailable or the ticker symbol is invalid. Please try: (1) Another ticker like AAPL or MSFT, (2) Waiting a few moments and trying again.'
            }), 400

        prediction = predictor.predict_next_day()

        if prediction is None:
            return jsonify({
                'success': False, 
                'error': 'Failed to generate prediction'
            }), 500

        print(f"Prediction successful for {ticker}!")
        return jsonify({
            'success': True,
            'prediction': prediction
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': f'Server error: Please try again'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Stock Predictor API'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
