from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
import traceback

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

regression_model = joblib.load('regression_model.pkl')
scaler_regression = joblib.load('scaler_regression.pkl')
classification_model = joblib.load('classification_model.pkl')
scaler_classification = joblib.load('scaler_classification.pkl')
clustering_model = joblib.load('clustering_model.pkl')
scaler_clustering = joblib.load('scaler_clustering.pkl')

df = pd.read_excel('AAPL.xlsx')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df.set_index('Date', inplace=True)

df['Daily_Return'] = df['Close'].pct_change()
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_10'] = df['Close'].rolling(window=10).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['Volatility'] = df['Daily_Return'].rolling(window=10).std()
df['High_Low_Spread'] = df['High'] - df['Low']
df['Volume_Change'] = df['Volume'].pct_change()
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))
df = df.dropna()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        print("Received price prediction request")
        data = request.json
        print(f"Request data: {data}")
        
        open_price = float(data['open'])
        high_price = float(data['high'])
        low_price = float(data['low'])
        close_price = float(data['close'])
        volume = float(data['volume'])
        
        print(f"Parsed values: O={open_price}, H={high_price}, L={low_price}, C={close_price}, V={volume}")
        
        last_close = df['Close'].iloc[-1]
        daily_return = (close_price - last_close) / last_close
        ma_5 = df['Close'].iloc[-5:].mean()
        ma_10 = df['Close'].iloc[-10:].mean()
        ma_20 = df['Close'].iloc[-20:].mean()
        volatility = df['Daily_Return'].iloc[-10:].std()
        high_low_spread = high_price - low_price
        volume_change = (volume - df['Volume'].iloc[-1]) / df['Volume'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        
        features = np.array([[open_price, high_price, low_price, close_price, volume, daily_return, ma_5, ma_10, ma_20, volatility, high_low_spread, volume_change, rsi]])
        features_scaled = scaler_regression.transform(features)
        predicted_price = regression_model.predict(features_scaled)[0]
        
        print(f"Prediction successful: {predicted_price}")
        
        return jsonify({
            'success': True,
            'predicted_price': round(predicted_price, 2),
            'current_price': round(close_price, 2),
            'price_change': round(predicted_price - close_price, 2),
            'percent_change': round(((predicted_price - close_price) / close_price) * 100, 2)
        })
    except Exception as e:
        print(f"Error in predict_price: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict_movement', methods=['POST'])
def predict_movement():
    try:
        print("Received movement prediction request")
        data = request.json
        print(f"Request data: {data}")
        
        daily_return = float(data['daily_return'])
        ma_5 = float(data['ma_5'])
        ma_10 = float(data['ma_10'])
        ma_20 = float(data['ma_20'])
        volatility = float(data['volatility'])
        high_low_spread = float(data['high_low_spread'])
        volume_change = float(data['volume_change'])
        rsi = float(data['rsi'])
        
        features = np.array([[daily_return, ma_5, ma_10, ma_20, volatility, high_low_spread, volume_change, rsi]])
        features_scaled = scaler_classification.transform(features)
        prediction = classification_model.predict(features_scaled)[0]
        probability = classification_model.predict_proba(features_scaled)[0]
        
        movement = "UP" if prediction == 1 else "DOWN"
        confidence = max(probability) * 100
        
        print(f"Prediction successful: {movement}")
        
        return jsonify({
            'success': True,
            'movement': movement,
            'confidence': round(confidence, 2),
            'probability_up': round(probability[1] * 100, 2),
            'probability_down': round(probability[0] * 100, 2)
        })
    except Exception as e:
        print(f"Error in predict_movement: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_cluster', methods=['POST'])
def get_cluster():
    try:
        print("Received cluster request")
        data = request.json
        print(f"Request data: {data}")
        
        mean_return = float(data['mean_return'])
        volatility = float(data['volatility'])
        avg_volume = float(data['avg_volume'])
        max_drawdown = float(data['max_drawdown'])
        
        features = np.array([[mean_return, volatility, avg_volume, max_drawdown]])
        features_scaled = scaler_clustering.transform(features)
        cluster = clustering_model.predict(features_scaled)[0]
        
        cluster_descriptions = {
            0: "Low Volatility - Stable Growth Period",
            1: "High Volatility - Turbulent Market Conditions",
            2: "Moderate - Correction or Consolidation Phase"
        }
        
        print(f"Cluster prediction successful: {cluster}")
        
        return jsonify({
            'success': True,
            'cluster': int(cluster),
            'description': cluster_descriptions.get(cluster, f"Cluster {cluster}"),
            'characteristics': {
                'mean_return': mean_return,
                'volatility': volatility,
                'avg_volume': avg_volume,
                'max_drawdown': max_drawdown
            }
        })
    except Exception as e:
        print(f"Error in get_cluster: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_historical_data', methods=['GET'])
def get_historical_data():
    try:
        last_30_days = df.tail(30)
        dates = last_30_days.index.strftime('%Y-%m-%d').tolist()
        prices = last_30_days['Close'].tolist()
        
        return jsonify({
            'success': True,
            'dates': dates,
            'prices': prices
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_latest_data', methods=['GET'])
def get_latest_data():
    try:
        latest = df.iloc[-1]
        return jsonify({
            'success': True,
            'date': latest.name.strftime('%Y-%m-%d'),
            'open': round(latest['Open'], 2),
            'high': round(latest['High'], 2),
            'low': round(latest['Low'], 2),
            'close': round(latest['Close'], 2),
            'volume': int(latest['Volume']),
            'daily_return': round(latest['Daily_Return'], 4),
            'ma_5': round(latest['MA_5'], 2),
            'ma_10': round(latest['MA_10'], 2),
            'ma_20': round(latest['MA_20'], 2),
            'volatility': round(latest['Volatility'], 4),
            'high_low_spread': round(latest['High_Low_Spread'], 2),
            'volume_change': round(latest['Volume_Change'], 4),
            'rsi': round(latest['RSI'], 2)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("FLASK SERVER STARTING...")
    print(f"Models loaded successfully:")
    print(f"Regression model")
    print(f"Classification model")
    print(f"Clustering model")
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nServer running at: http://127.0.0.1:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')