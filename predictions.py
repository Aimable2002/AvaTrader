import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import MetaTrader5 as mt5
from utilis import save_prediction_results, update_prediction_outcomes
from model import ForexLSTM
from update import load_model
from plNews import estimate_sentiment
# Dataset class to handle data in PyTorch format
class ForexDataset(Dataset):
    def __init__(self, X, y):
        # Convert numpy arrays to PyTorch tensors
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)  # Reshape to ensure correct dimensions
    
    def __len__(self):
        return len(self.X)  # Return the size of the dataset
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # Return a single sample and label

# LSTM Neural Network Model


def calculate_rsi(prices, periods=10):
    """
    Calculate Relative Strength Index (RSI)
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    # Calculate price changes
    delta = prices.diff()
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    # Calculate RS and RSI
    rs = gain / loss
    return 100 - (100 / (1 + rs))



def calculate_atr(high, low, close, window=10):
    """
    Calculate Average True Range (ATR)
    TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    ATR = Moving average of TR
    """
    # Calculate the three differences
    tr1 = high - low  # Current high - current low
    tr2 = abs(high - close.shift())  # Current high - previous close
    tr3 = abs(low - close.shift())  # Current low - previous close
    # Get the maximum of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Calculate moving average
    return tr.rolling(window=window).mean()

def prepare_forex_data(df, symbol, sequence_length=10):
    """
    Prepare forex data with technical indicators for prediction
    Args:
        df: DataFrame with forex data
        symbol: Currency pair symbol
        sequence_length: Number of time steps to use for prediction
        sentiment_value: Sentiment value (1 for positive, -1 for negative, 0 for neutral)
        sentiment_probability: Sentiment probability (0 to 1)
    """
    # Select base features
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
    data = df[symbol][features].copy()
    
    # Add technical indicators
    data['MA_5'] = data['close'].rolling(window=5).mean()  # 5-day Moving Average
    data['MA_10'] = data['close'].rolling(window=10).mean()  # 10-day Moving Average
    data['RSI'] = calculate_rsi(data['close'], periods=10)  # RSI
    data['ATR'] = calculate_atr(data['high'], data['low'], data['close'], window=10)  # ATR
    
    # Add price momentum indicators
    data['Price_Change'] = data['close'].pct_change()  # 1-period price change
    data['Price_Change_5'] = data['close'].pct_change(periods=5)  # 5-period price change
    data['Price_Change_10'] = data['close'].pct_change(periods=10)  # 10-period price change
    
    # if sentiment_value is not None and sentiment_probability > 0.7:
    #     data['Sentiment_Value'] = sentiment_value
    #     data['Sentiment_Probability'] = sentiment_probability

    # Remove NaN values
    data = data.dropna()
    
    # Limit to 1000 rows if more data is available
    if len(data) > 1000:
        data = data.tail(1000)
    
    # Scale the data between 0 and 1
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])  # Input sequence
        y.append(scaled_data[i + sequence_length, 3])   # Target (next close price)
    
    print(f"Input data shape: {(np.array(X)).shape}")

    return np.array(X), np.array(y), scaler


# ... (after the train_model function) =ticker
#ticker = ["EURUSD", "EURGBP", "EURCAD","GBPJPY", "EURJPY", "EURAUD", "EURCHF", "AUDUSD", "GBPUSD", "USDCHF", "USDJPY"]
def predict_next_prices(df, symbols):
    """
    Make predictions for multiple forex symbols
    Args:
        df: DataFrame with historical data
        symbols: List of forex symbols to predict
    Returns:
        dict: Predictions for each symbol
    """
    predictions = {}
    # print(f"\n df in predict_next_prices: {df.iloc[-1]}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for symbol in symbols:
        try:
            print(f"\nProcessing {symbol}...")

            # Create a copy of the current data
            current_data = df[symbol].copy()
            save_training_data(current_data, symbol)

            # Save current training data
            # save_training_data(df[symbol], symbol)
            
            # Load and combine with historical training data
            historical_data = load_and_combine_training_data(symbol)
            # print(f"\n df in predict_next_prices historical data: {historical_data.iloc[-1]}")
            if historical_data is not None:
                # df[symbol] = pd.concat([historical_data, df[symbol]]).drop_duplicates()
                # print(f"Combined with historical training data for {symbol}")

                # Ensure both DataFrames have the same columns
                # print(f"\n df in predict_next_prices current data: {current_data.iloc[-1]}")
                combined_data = pd.concat([historical_data, current_data], axis=0)
                # combined_data = combined_data.drop_duplicates().reset_index(drop=True)
                combined_data = combined_data.drop_duplicates(subset=['time'], keep='last').reset_index(drop=True)
                combined_data = combined_data.astype(df[symbol].dtypes.to_dict())
                # print(f"Combined data for {symbol}: {combined_data.iloc[-1]}") 
                # print(f"Combined data for {symbol} (last 5 rows):\n{combined_data.tail()}")
                # print(f'\n df symbol: {df[symbol].iloc[-1]}')
                # df[symbol] = combined_data
                # df.loc[:, symbol] = combined_data
                # print(f"\n df in predict_next_prices combined data: {df[symbol].iloc[-1]}")
                # print(f"Updated df for {symbol} (last 5 rows):\n{df[symbol].tail()}")
                # print(f"Combined with historical training data for {symbol}")
                
            # Prepare data with 10-day sequences
            # probability, sentiment = estimate_sentiment(symbol)
            # print(f"\n In prediction Price Sentiment for {symbol}: {sentiment}")
            # print(f"\n In prediction Price Confidence: {probability:.2%}")
            # sentiment_value = 0
            # sentiment_probability = 0

            # # Only consider sentiment if probability is greater than 0.7
            # if probability > 0.7:
            #     sentiment_value = 1 if sentiment == "positive" else -1 if sentiment == "negative" else 0
            #     sentiment_probability = probability         sentiment_value=sentiment_value, sentiment_probability=sentiment_probability

            X, y, scaler = prepare_forex_data(df, symbol, sequence_length=10)
            
            if len(X) < 20:  # Minimum data check
                print(f"Not enough data for {symbol}")
                continue
            
            # Split data (80% train, 20% validation)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=False  # No shuffle for time series
            )
            
            # Create datasets and dataloaders
            train_dataset = ForexDataset(X_train, y_train)
            val_dataset = ForexDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Get input size (number of features)
            input_size = X.shape[2]

            # Initialize and train model
            # model = ForexLSTM(input_size=input_size)
            
            # Try to load existing model first
            model = load_model(symbol, input_size)
            if model is None:
                # Initialize and train new model if no valid saved model exists
                model = ForexLSTM(input_size=input_size)
            model = train_model(model, train_loader, val_loader, symbol)
            
            # Make prediction
            model.eval()
            with torch.no_grad():
                # Use last 10 days for prediction
                last_sequence = torch.FloatTensor(X[-1:]).to(device)
                predicted_scaled = model(last_sequence).unsqueeze(-1)
                
                # Inverse transform
                dummy = np.zeros((1, input_size))
                dummy[:, 3] = predicted_scaled.cpu().numpy().flatten()
                predicted_price = scaler.inverse_transform(dummy)[0, 3]
                
                current_price = df[symbol]['close'].iloc[-1]
                # print(f"\n Print the last df row: {df[symbol].iloc[-1]}")
                predictions[symbol] = {
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'movement': 'UP' if predicted_price > current_price else 'DOWN',
                    'change_percent': ((predicted_price - current_price) / current_price) * 100
                }
                
                print(f"\nPrediction for {symbol}:")
                print(f"Current price: {current_price:.5f}")
                print(f"Predicted next price: {predicted_price:.5f}")
                print(f"Predicted movement: {predictions[symbol]['movement']}")
                print(f"Predicted change: {predictions[symbol]['change_percent']:.2f}%")

                # Update outcomes for previous predictions
                update_prediction_outcomes(symbol)

            # make_trade_decision(predicted_price, stop_loss, take_profit, threshold=1.5)

        except Exception as e:
            print(f"Error predicting {symbol}: {str(e)}")
            continue
    
    return predictions



def save_training_data(df, symbol, max_records=10000):
    """
    Save training data to CSV file
    """
    try:
        import os
        from datetime import datetime
        
        # Create data directory if it doesn't exist
        if not os.path.exists('training_data'):
            os.makedirs('training_data')
            
        # Save data with timestamp
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f'training_data/{symbol}_{timestamp}.csv'
        df.to_csv(filename)
        print(f"\n df in save_training_data: {df.iloc[-1]}")
        print(f"Training data saved to {filename}")
        
    except Exception as e:
        print(f"Error saving training data: {e}")

def load_and_combine_training_data(symbol, days_to_keep=30):
    """
    Load and combine historical training data
    """
    try:
        import os
        import glob
        from datetime import datetime, timedelta
        
        # Get all files for this symbol
        files = glob.glob(f'training_data/{symbol}_*.csv')
        
        if not files:
            return None
        
        # Filter files by date
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        recent_files = []
        
        for file in files:
            # Extract date from filename
            date_str = file.split('_')[-1].replace('.csv', '')
            file_date = datetime.strptime(date_str, '%Y%m%d')
            
            if file_date >= cutoff_date:
                recent_files.append(file)
        
        if not recent_files:
            return None
        
        dfs = []
        for f in recent_files:
            df = pd.read_csv(f, index_col=0)  # Add index_col=0 to use first column as index
            dfs.append(df)
        
        # Combine all recent files
        # combined_data = pd.concat([pd.read_csv(f) for f in recent_files])
        # return combined_data

        combined_data = pd.concat(dfs)
        combined_data = combined_data.reset_index(drop=True)  # Reset index after combining
        return combined_data
        
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None
    


def train_model(model, train_loader, val_loader, symbol, epochs=100, learning_rate=0.001):
    """
    Train the LSTM model
    Args:
        model: LSTM model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        symbol: Forex symbol being trained
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Correct import for ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.75, 
        patience=10,
        min_lr=0.0001
        # verbose=True
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 20  # Early stopping patience
    patience_counter = 0
    min_delta = 0.0001 
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        batch_count = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).squeeze(-1)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            batch_count += 1

            if batch_count % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_count}: Loss = {loss.item():.6f}")
        
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device).squeeze(-1)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()

                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

            
            # Add model evaluation here
            directional_accuracy, sharpe_ratio = evaluate_model(val_predictions, val_targets)
            print(f"\nEpoch {epoch+1} Evaluation Metrics:")
            print(f"Directional Accuracy: {directional_accuracy:.2f}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)


        val_accuracy = np.mean(np.sign(np.array(val_predictions)) == np.sign(np.array(val_targets)))
        
        print(f'\nEpoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Val Loss: {avg_val_loss:.6f}')
        print(f'Val Accuracy: {val_accuracy:.2f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            print(f'Best validation loss: {best_val_loss:.6f}')
            break
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {avg_train_loss:.6f}')
            print(f'Val Loss: {avg_val_loss:.6f}')
            print(f'Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Add this: Save the trained model
        try:
            save_model(model, symbol)  # You'll need to pass the symbol parameter
        except Exception as e:
            print(f"Error saving model: {e}")
    
    return model

def save_model(model, symbol):
    """Save trained model and its metadata to disk"""
    try:
        import os
        import json
        from datetime import datetime
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save model state
        model_path = f'models/{symbol}_model.pth'
        torch.save(model.state_dict(), model_path)
        
        # Save model metadata
        metadata = {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input_size': model.lstm.input_size,
            'hidden_size': model.lstm.hidden_size,
            'num_layers': model.lstm.num_layers
        }
        
        metadata_path = f'models/{symbol}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")
        
    except Exception as e:
        print(f"Error saving model: {e}")





def retrain_model_with_feedback(symbol, model, recent_predictions, sequence_length=10, learning_rate=0.001):
    """
    Retrain the model using recent prediction outcomes
    """
    try:
        successful_sequences = []
        successful_targets = []
        
        # Filter predictions with actual outcomes
        completed_predictions = [p for p in recent_predictions if p.get('actual_outcome')]
        
        if len(completed_predictions) < 50:  # Need minimum samples
            return model
            
        # Load recent market data
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1000)
        if rates is None:
            return model
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Prepare data
        X, y, scaler = prepare_forex_data(df, symbol, sequence_length)
        
        # Add weight to successful predictions
        weights = []
        for pred in completed_predictions:
            if pred['predicted_direction'] == pred['actual_outcome']['movement']:
                weights.append(1.5)  # Higher weight for correct predictions
                if pred['actual_outcome']['profit'] > 0:
                    weights.append(2.0)
            else:
                weights.append(0.5)  # Lower weight for incorrect predictions
        
        # Create weighted dataset
        weighted_dataset = ForexDataset(X, y)
        weighted_sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            weighted_dataset,
            batch_size=32,
            sampler=weighted_sampler
        )
        
        # Retrain model
        model = train_model(model, train_loader, train_loader, symbol, epochs=50)
        
        # Retrain model with adaptive learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=train_loader,  # Using same loader for validation
            symbol=symbol,
            epochs=50,
            optimizer=optimizer
        )
        # Save retrained model
        save_model(model, symbol)
        print(f"Model retrained for {symbol} with feedback from {len(completed_predictions)} predictions")
        
        return model
        
    except Exception as e:
        print(f"Error retraining model: {e}")
        return model





def evaluate_model(predictions, actuals):
    """Evaluate model using trading-specific metrics."""
    # Directional accuracy
    correct_direction = sum((pred > 0) == (act > 0) for pred, act in zip(predictions, actuals))
    directional_accuracy = correct_direction / len(predictions)
    
    # Calculate Sharpe ratio
    returns = [act - pred for pred, act in zip(predictions, actuals)]
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe_ratio = mean_return / std_return if std_return != 0 else 0
    
    print(f"Directional Accuracy: {directional_accuracy:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    return directional_accuracy, sharpe_ratio


# def track_mistake_patterns(predictions):
#     """Analyze patterns in incorrect predictions"""
#     mistake_patterns = {}
#     for pred in predictions:
#         if pred['predicted_direction'] != pred['actual_outcome']['movement']:
#             # Record market conditions during mistake
#             conditions = extract_market_conditions(pred)
#             pattern_key = str(conditions)
#             mistake_patterns[pattern_key] = mistake_patterns.get(pattern_key, 0) + 1
#     return mistake_patterns