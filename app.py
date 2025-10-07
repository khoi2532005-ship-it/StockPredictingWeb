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
warnings.filterwarnings('ignore')

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Configure yfinance session with timeout and user agent
import requests
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

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
        self.vix_symbol = "^VIX"

    def load_data(self):
        """Load and prepare stock data"""
        try:
            # Calculate date range - last 300 days from today
            end_date = datetime.now()
            start_date = end_date - timedelta(days=300)

            print(f"Loading data for {self.ticker_symbol} from {start_date.date()} to {end_date.date()}")

            # Download stock data with timeout and session
            try:
                data = yf.download(
                    tickers=[self.ticker_symbol, self.vix_symbol], 
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    progress=False,
                    timeout=30
                )
            except Exception as download_error:
                print(f"Error downloading data: {download_error}")
                # Try downloading without VIX as fallback
                data = yf.download(
                    tickers=self.ticker_symbol, 
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    progress=False,
                    timeout=30
                )

            if data.empty:
                print(f"Error: No data found for {self.ticker_symbol}")
                return False

            # Handle single ticker vs multiple tickers data structure
            if 'Close' in data.columns:
                if isinstance(data['Close'], pd.DataFrame):
                    close = data["Close"]
                else:
                    # Single ticker - create DataFrame
                    close = pd.DataFrame({self.ticker_symbol: data['Close']})
            else:
                print(f"Error: Unexpected data structure for {self.ticker_symbol}")
                return False

            # Check if we have enough data
            if len(close) < 30:
                print(f"Error: Insufficient data for {self.ticker_symbol} (only {len(close)} days)")
                return False

            # Create features for Ridge model
            stock_features = pd.DataFrame({
                'stock_sma5': close[self.ticker_symbol].rolling(5).mean().shift(1),
                'stock_sma20': close[self.ticker_symbol].rolling(20).mean().shift(1)
            })

            # Check if VIX data is available
            if self.vix_symbol in close.columns:
                vix_features = pd.DataFrame({
                    'vix_sma5': close[self.vix_symbol].rolling(5).mean().shift(1),
                    'vix_sma20': close[self.vix_symbol].rolling(20).mean().shift(1)
                })
                features = pd.concat([stock_features, vix_features], axis=1).dropna()
            else:
                features = stock_features.dropna()

            target = close[self.ticker_symbol].loc[features.index]

            # Scale features for Ridge
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(features)
            y = target.values

            # Train Ridge model
            self.ridge_model = Ridge(solver='svd', alpha=1, random_state=69)
            self.ridge_model.fit(X_scaled, y)

            # Prepare LSTM data
            lookback = 20
            X_lstm, y_lstm = [], []

            stock_close = close[self.ticker_symbol].dropna()

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

            self.close_data = close
            self.features = features
            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict_next_day(self):
        """Predict next trading day price"""
        try:
            if not all([self.ridge_model, self.lstm_model, self.scaler, 
                       self.scaler_x, self.scaler_y]):
                return None

            last_date = self.close_data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                last_date = last_date.to_pydatetime()

            current_price = self.close_data[self.ticker_symbol].iloc[-1]

            next_date = last_date + timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)

            stock_sma5 = self.close_data[self.ticker_symbol].rolling(5).mean().iloc[-1]
            stock_sma20 = self.close_data[self.ticker_symbol].rolling(20).mean().iloc[-1]

            if self.vix_symbol in self.close_data.columns:
                vix_sma5 = self.close_data[self.vix_symbol].rolling(5).mean().iloc[-1]
                vix_sma20 = self.close_data[self.vix_symbol].rolling(20).mean().iloc[-1]
                feature_row = pd.DataFrame([[stock_sma5, stock_sma20, vix_sma5, vix_sma20]], 
                                         columns=self.features.columns)
            else:
                feature_row = pd.DataFrame([[stock_sma5, stock_sma20]], 
                                         columns=self.features.columns)

            feature_scaled = self.scaler.transform(feature_row)
            ridge_prediction = self.ridge_model.predict(feature_scaled)[0]

            stock_close = self.close_data[self.ticker_symbol].dropna()
            last_sequence = stock_close.iloc[-20:].values.reshape(1, 20, 1)
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

        print(f"Predicting for ticker: {ticker}")

        predictor = StockPredictor(ticker)

        if not predictor.load_data():
            return jsonify({
                'success': False, 
                'error': f'Failed to load data for {ticker}. This could be due to an invalid ticker symbol or temporary connection issues with Yahoo Finance. Please try again.'
            }), 400

        prediction = predictor.predict_next_day()

        if prediction is None:
            return jsonify({
                'success': False, 
                'error': 'Failed to make prediction'
            }), 500

        return jsonify({
            'success': True,
            'prediction': prediction
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
