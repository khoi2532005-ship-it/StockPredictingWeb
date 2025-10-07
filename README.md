# 📈 Stock Price Predictor - Flask Web Application

AI-powered stock price prediction using Ridge Regression and LSTM Neural Networks.

## 🚀 Quick Start

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python app.py
   ```

3. **Open browser**: `http://localhost:5000`

### Deploy to Heroku

#### Option 1: Via GitHub (Recommended)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
   git push -u origin main
   ```

2. **Connect to Heroku**:
   - Go to [Heroku Dashboard](https://dashboard.heroku.com)
   - Create new app
   - Go to Deploy tab → GitHub → Connect repository
   - Enable Automatic Deploys or Deploy Branch

#### Option 2: Via Heroku CLI

1. **Login to Heroku**:
   ```bash
   heroku login
   ```

2. **Create and deploy**:
   ```bash
   heroku create your-app-name
   git push heroku main
   heroku open
   ```

## 📁 Project Structure

```
stock-predictor/
├── app.py              # Flask application
├── Procfile            # Heroku startup command
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python version
├── templates/
│   └── index.html      # Web interface
├── .gitignore          # Excluded files
└── README.md           # Documentation
```

## 🎯 Features

- **Type Any Ticker**: Enter any stock symbol (AAPL, TSLA, GOOGL, etc.)
- **Dual AI Models**: Ridge Regression + LSTM Neural Network
- **Real-time Data**: Last 300 days from Yahoo Finance
- **Model Agreement**: See when both models align
- **Mobile Friendly**: Responsive design

## 🛠️ Tech Stack

- **Backend**: Flask, Python 3.11
- **ML Models**: scikit-learn (Ridge), TensorFlow (LSTM)
- **Data**: yfinance (Yahoo Finance API)
- **Deployment**: Heroku with Gunicorn

## 📊 API Endpoints

### `POST /predict`
Get stock price prediction

**Request**:
```json
{
  "ticker": "AAPL"
}
```

**Response**:
```json
{
  "success": true,
  "prediction": {
    "ticker": "AAPL",
    "current_price": 178.50,
    "ridge_prediction": 179.25,
    "lstm_prediction": 178.95,
    "average_prediction": 179.10,
    "models_agree": true
  }
}
```

### `GET /health`
Health check endpoint

## ⚠️ Disclaimer

This application is for educational purposes only. Stock predictions are uncertain and should not be the sole basis for investment decisions.

## 📄 License

MIT License - Free to use and modify.

---

**Made with ❤️ using Flask, TensorFlow, and scikit-learn**
