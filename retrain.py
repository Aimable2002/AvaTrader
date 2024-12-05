import os
import json
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import MetaTrader5 as mt5
from model import ForexLSTM
from learningMonitor import LearningMonitor
from utilis import update_prediction_outcomes
import traceback

class RetrainBot:
    def __init__(self):
        self.model = ForexLSTM()
        self.learning_monitor = LearningMonitor()
        self.last_retrain_time = datetime.now()
        self.retrain_interval = timedelta(hours=24)  # Retrain daily
        self.min_accuracy_threshold = 55  # 55% minimum accuracy
        self.min_profit_threshold = 0  # Minimum profit threshold

    def prepare_training_data(self, training_data):
        """Convert historical trades to training data"""
        try:
            X = []  # Features
            y = []  # Labels
            sequence_length = 10  # Number of time steps to look back
            
            for trade in training_data:
                # Get historical price data for this trade
                symbol = trade['symbol']
                end_time = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S')
                start_time = end_time - timedelta(minutes=5 * sequence_length)
                
                # Get OHLCV data
                rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_time, end_time)
                if rates is None or len(rates) < sequence_length:
                    continue
                    
                # Prepare features (last 10 candles)
                features = []
                for rate in rates[-sequence_length:]:
                    candle_features = [
                        rate['open'],
                        rate['high'],
                        rate['low'],
                        rate['close'],
                        rate['tick_volume']
                    ]
                    features.append(candle_features)
                
                # Prepare label (actual movement)
                if trade.get('actual_outcome'):
                    label = 1 if trade['actual_outcome']['movement'] == 'UP' else 0
                else:
                    continue
                    
                X.append(features)
                y.append(label)
            
            if not X or not y:
                print("No valid training data found")
                return None, None
                
            # Convert to tensors
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            
            print(f"Prepared {len(X)} training samples")
            return X, y
            
        except Exception as e:
            print(f"Error preparing training data: {e}")
            return None, None

    def retrain_model(self):
        """Retrain model using historical data and mistakes"""
        try:
            print("\nStarting model retraining...")
            
            # 1. Collect training data
            training_data = []
            for symbol in self.ticker:
                filename = f'prediction_history/{symbol}_predictions.json'
                if not os.path.exists(filename):
                    continue
                    
                with open(filename, 'r') as f:
                    predictions = json.load(f)
                
                # Filter completed predictions
                completed = [p for p in predictions if p.get('actual_outcome')]
                training_data.extend(completed)
            
            if not training_data:
                print("No training data available")
                return
            
            # 2. Prepare data
            X_train, y_train = self.prepare_training_data(training_data)
            if X_train is None or y_train is None:
                return
                
            # 3. Set up training
            self.model.train()  # Set to training mode
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.BCEWithLogitsLoss()
            batch_size = 32
            n_epochs = 10
            
            # 4. Training loop
            print("\nStarting training loop...")
            for epoch in range(n_epochs):
                total_loss = 0
                n_batches = 0
                
                # Create batches
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    n_batches += 1
                
                avg_loss = total_loss / n_batches
                print(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")
                
            # 5. Save updated model
            model_path = 'models/forex_lstm.pth'
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            
            # 6. Reset to evaluation mode
            self.model.eval()
            
            # 7. Update learning monitor
            self.learning_monitor.record_training_session({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'samples_trained': len(X_train),
                'final_loss': avg_loss,
                'epochs': n_epochs
            })
            
            print("Model retraining completed successfully")
            
        except Exception as e:
            print(f"Error retraining model: {e}")
            print("Stack trace:", traceback.format_exc())

    def analyze_performance(self):
        """Analyze trading performance across all symbols"""
        try:
            total_accuracy = 0
            total_profit = 0
            symbols_analyzed = 0
            
            for symbol in self.ticker:
                filename = f'prediction_history/{symbol}_predictions.json'
                if not os.path.exists(filename):
                    continue
                    
                with open(filename, 'r') as f:
                    predictions = json.load(f)
                
                # Filter completed predictions
                completed = [p for p in predictions if p.get('actual_outcome')]
                if not completed:
                    continue
                    
                # Calculate metrics
                correct = sum(1 for p in completed 
                            if p['predicted_direction'] == p['actual_outcome']['movement'])
                accuracy = (correct / len(completed)) * 100
                profit = sum(p['actual_outcome']['profit'] for p in completed)
                
                total_accuracy += accuracy
                total_profit += profit
                symbols_analyzed += 1
                
                # Track in learning monitor
                self.learning_monitor.track_mistake_patterns(completed, symbol)
                
            if symbols_analyzed > 0:
                avg_accuracy = total_accuracy / symbols_analyzed
                print(f"\nOverall Performance:")
                print(f"Average Accuracy: {avg_accuracy:.2f}%")
                print(f"Total Profit: {total_profit:.2f}")
                
                return {
                    'accuracy': avg_accuracy,
                    'total_profit': total_profit,
                    'symbols_analyzed': symbols_analyzed
                }
            
            return None
            
        except Exception as e:
            print(f"Error analyzing performance: {e}")
            return None

    def should_retrain(self, performance_metrics):
        """Determine if retraining is needed"""
        if not performance_metrics:
            return False
            
        return (
            performance_metrics['accuracy'] < self.min_accuracy_threshold or
            performance_metrics['total_profit'] < self.min_profit_threshold
        )

    def learning_cycle(self):
        try:
            # 1. Check if it's time to analyze and retrain
            if datetime.now() - self.last_retrain_time > self.retrain_interval:
                print("\nStarting learning cycle...")
                
                # 2. Update outcomes for all symbols
                for symbol in self.ticker:
                    update_prediction_outcomes(symbol)
                
                # 3. Analyze performance
                performance_metrics = self.analyze_performance()
                
                # 4. Check if retraining is needed
                if self.should_retrain(performance_metrics):
                    print("Performance below threshold, initiating retraining...")
                    self.retrain_model()
                
                self.last_retrain_time = datetime.now()
                
        except Exception as e:
            print(f"Error in learning cycle: {e}")