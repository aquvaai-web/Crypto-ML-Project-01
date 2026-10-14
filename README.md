# CryptoVision – ML Crypto Market Analyzer

**CryptoVision** is a full-stack machine learning project built for cryptocurrency market trend classification, technical analysis indicator calculation, computer-vision candlestick chart feature extraction, and ML explainability. It is designed specifically for college ML assignments and viva defense.

---

## Architecture Diagram

```
User Uploads Candlestick Chart Image / Enter Market Parameters
                           ↓
        Computer Vision Processing (OpenCV / Pillow)
        (Candlestick Pixel Color Ratio & Edge Density)
                           ↓
        Technical Indicators Engine (Pandas / NumPy)
   (SMA 20/50, EMA 20/50/200, RSI 14, MACD, Bollinger Bands)
                           ↓
        Anti-Leakage Chronological Feature Vector (80/20)
                           ↓
    ML Models Evaluation (Logistic Regression vs Random Forest vs XGBoost)
                           ↓
   Multi-Factor Signal Scoring & Hypothetical Trade Setup Engine
                           ↓
             FastAPI REST Backend (`http://127.0.0.1:8000`)
                           ↓
      Modern Dark Trading Dashboard Frontend (`http://127.0.0.1:5173`)
```

---

## Key Features

1. **Anti-Leakage Financial ML Pipeline**:
   - Uses strict **chronological train/test splitting** (80% historical past, 20% future test data).
   - Prevents lookahead bias by calculating indicators using only data up to timestamp $t$.
   - Trains and compares **Logistic Regression**, **Random Forest**, and **XGBoost**.

2. **Computer Vision Candlestick Chart Analysis**:
   - Employs OpenCV HSV color filtering to detect green (bullish) vs red (bearish) candlestick body pixel ratios.
   - Computes Canny edge density to evaluate chart volatility structure.

3. **Multi-Factor Market Trend & Trade Setup Engine**:
   - Transparent scoring combining ML model vote, RSI momentum, MACD crossovers, EMA trend alignment, and CV chart visuals.
   - Generates model-derived hypothetical trade setups (Entry Zone, Stop Loss, Target 1, Target 2, Risk/Reward Ratio).

4. **ML Model Explainability**:
   - Displays feature importance bar chart showing top predictors (EMA 20, EMA 200, RSI 14, MACD).

5. **College Viva Defense Section**:
   - Includes dedicated "How the ML model works" guide explaining key concepts (Target variable, avoiding data leakage, ensemble model choice).

---

## How to Run Locally

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Set Up & Train ML Backend

```bash
# Navigate to project root directory
cd "Crypto ML Project"

# Install Python dependencies
pip install -r requirements.txt

# Generate dataset & train ML models (Logistic Regression, Random Forest, XGBoost)
python backend/train_models.py

# Start FastAPI backend server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Backend API will run at `http://127.0.0.1:8000` (Swagger docs available at `http://127.0.0.1:8000/docs`).

### 2. Set Up & Launch Frontend Dashboard

In a new terminal window:

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Vite dev server
npx vite --host 127.0.0.1 --port 5173
```
Open `http://127.0.0.1:5173` in your browser.

---

## Project Structure

```
Crypto ML Project/
├── backend/
│   ├── dataset/
│   │   └── btc_usdt_ohlcv.csv         # 2-year OHLCV dataset
│   ├── models/
│   │   ├── best_model.pkl             # Trained best model (XGBoost)
│   │   └── metrics.json               # Logged evaluation metrics
│   ├── data_processor.py              # Indicator calculations & feature engineering
│   ├── generate_dataset.py            # OHLCV synthetic data generator
│   ├── image_analyzer.py              # OpenCV computer vision chart processor
│   ├── predictor.py                   # Multi-factor signal & trade setup engine
│   ├── train_models.py                # Chronological split ML model trainer
│   └── main.py                        # FastAPI web server
├── frontend/                          # React + Vite application
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── ImageUploader.jsx
│   │   │   ├── MarketOutlookCard.jsx
│   │   │   ├── TradeSetupCard.jsx
│   │   │   ├── TechnicalIndicatorsCard.jsx
│   │   │   ├── ModelExplainabilityCard.jsx
│   │   │   ├── ModelPerformance.jsx
│   │   │   └── AboutPage.jsx
│   │   ├── App.jsx
│   │   └── index.css                  # Dark theme trading UI system
│   ├── package.json
│   └── vite.config.js
├── generate_sample_chart.py           # Script to generate sample candlestick chart image
├── sample_chart.png                   # Sample chart image for testing upload
├── requirements.txt                   # Python package dependencies
└── README.md                          # Project documentation & viva guide
```

---

## Viva Defense Questions & Answers

1. **What is the Target Variable?**
   - Binary classification: `1` for Bullish movement (future 5-day return > +0.5%) and `0` for Bearish/Sideways.

2. **Why is chronological splitting required instead of K-Fold cross-validation?**
   - Financial time-series data possesses temporal autocorrelation. Standard random K-Fold shuffling leaks future price information into training samples. Chronological splitting (80% past, 20% future) preserves temporal integrity.

3. **How does OpenCV process chart images?**
   - Converts the uploaded candlestick image into HSV color space, isolates green (`#10b981`) and red (`#ef4444`) pixel regions to determine candlestick body ratio, and calculates Canny edge density for volatility structure.

---

## Disclaimer
*CryptoVision is developed strictly for educational and college assignment purposes. All trade setups and predictions are model-generated hypothetical simulations and should not be used as financial trading advice.*
