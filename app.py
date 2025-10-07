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
warnings.filterwarnings('ignore')

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

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
        """Load and prepare stock data with enhanced error handling"""
        try:
            # Calculate date range - last 300 days from today
            end_date = datetime.now()
            start_date = end_date - timedelta(days=300)

            print(f"Loading data for {self.ticker_symbol} from {start_date.date()} to {end_date.date()}")

            # Try downloading with retries and better error handling
            max_retries = 3
            data = None

            for attempt in range(max_retries):
                try:
                    # Create Ticker object
                    ticker = yf.Ticker(self.ticker_symbol)

                    # Download historical data
                    data = ticker.history(
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        interval='1d'
                    )

                    if not data.empty:
                        print(f"Successfully downloaded {len(data)} days of data for {self.ticker_symbol}")
                        break

                    time.sleep(1)  # Wait before retry

                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    continue

            if data is None or data.empty:
                print(f"Error: No data found for {self.ticker_symbol} after {max_retries} attempts")
                return False

            # Use Close price column
            if 'Close' not in data.columns:
                print(f"Error: No Close price data for {self.ticker_symbol}")
                return False

            stock_close = data['Close'].dropna()

            # Check if we have enough data
            if len(stock_close) < 30:
                print(f"Error: Insufficient data for {self.ticker_symbol} (only {len(stock_close)} days)")
                return False

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

        print(f"Predicting for ticker: {ticker}")

        predictor = StockPredictor(ticker)

        if not predictor.load_data():
            return jsonify({
                'success': False, 
                'error': f'Unable to fetch data for {ticker}. This may be due to: (1) Invalid ticker symbol, (2) Temporary connection issues, or (3) Data provider limitations. Please try again or use a different ticker.'
            }), 400

        prediction = predictor.predict_next_day()

        if prediction is None:
            return jsonify({
                'success': False, 
                'error': 'Failed to generate prediction'
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
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
