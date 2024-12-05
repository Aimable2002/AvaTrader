# import MetaTrader5 as mt5
# import json
# from datetime import datetime, timedelta
# import os
# from predictions import load_model
# import numpy as np
# from learningMonitor import LearningMonitor
# from utilis import save_prediction_results, update_prediction_outcomes, analyze_prediction_accuracy, load_model

# learning_monitor = LearningMonitor()


# def save_prediction_results(symbol, prediction_data, actual_outcome=None):
#     """
#     Save prediction results and actual outcomes for model evaluation
#     Args:
#         symbol: Trading symbol
#         prediction_data: Dictionary containing prediction details
#         actual_outcome: Actual price movement (to be updated later)
#     """
#     try:
        
#         # Create predictions directory if it doesn't exist
#         if not os.path.exists('prediction_history'):
#             os.makedirs('prediction_history')
            
#         # Prepare prediction record
#         record = {
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'current_price': prediction_data['current_price'],
#             'predicted_price': prediction_data['predicted_price'],
#             'predicted_movement': prediction_data['movement'],
#             'change_percent': prediction_data['change_percent'],
#             'actual_outcome': actual_outcome
#         }
        
#         # Save to JSON file
#         filename = f'prediction_history/{symbol}_predictions.json'
        
#         # Load existing predictions if file exists
#         existing_predictions = []
#         if os.path.exists(filename):
#             with open(filename, 'r') as f:
#                 existing_predictions = json.load(f)
                
#         # Add new prediction
#         existing_predictions.append(record)
        
#         # Keep only last 1000 predictions
#         if len(existing_predictions) > 1000:
#             existing_predictions = existing_predictions[-1000:]
            
#         # Save updated predictions
#         with open(filename, 'w') as f:
#             json.dump(existing_predictions, f, indent=4)
            
#         print(f"Prediction saved to {filename}")
        
#     except Exception as e:
#         print(f"Error saving prediction: {e}")

# def update_prediction_outcomes(symbol, timeframe=1):
#     """
#     Update previous predictions with actual outcomes
#     Args:
#         symbol: Trading symbol
#         timeframe: Number of periods to look back for verification
#     """
#     try:
        
        
#         filename = f'prediction_history/{symbol}_predictions.json'
#         if not os.path.exists(filename):
#             return
            
#         # Load predictions
#         with open(filename, 'r') as f:
#             predictions = json.load(f)
            
#         # Get current price from MT5
#         current_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
#         if current_rates is None or len(current_rates) == 0:
#             return
            
#         current_price = current_rates[0]['close']
        
#         # Update predictions that don't have actual outcomes
#         updated = False
#         for pred in predictions:
#             if pred['actual_outcome'] is None:
#                 pred_time = datetime.strptime(pred['timestamp'], '%Y-%m-%d %H:%M:%S')
#                 if datetime.now() - pred_time > timedelta(minutes=timeframe):
#                     # Calculate actual outcome
#                     actual_change = ((current_price - pred['current_price']) / 
#                                    pred['current_price']) * 100
#                     pred['actual_outcome'] = {
#                         'price': current_price,
#                         'change_percent': actual_change,
#                         'movement': 'UP' if actual_change > 0 else 'DOWN'
#                     }
#                     updated = True
                    
#         if updated:
#             # Save updated predictions
#             with open(filename, 'w') as f:
#                 json.dump(predictions, f, indent=4)
                
#             # Analyze prediction accuracy
#             analyze_prediction_accuracy(symbol)
            
#     except Exception as e:
#         print(f"Error updating prediction outcomes: {e}")

from model import ForexLSTM
import torch  

def load_model(symbol, input_size):
    """Load trained model from disk if it exists"""
    try:
        import os
        import json
        from datetime import datetime, timedelta
        
        model_path = f'models/{symbol}_model.pth'
        metadata_path = f'models/{symbol}_metadata.json'
        
        # Check if model and metadata exist
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if model is not too old (e.g., less than 1 day old)
            saved_time = datetime.strptime(metadata['timestamp'], '%Y-%m-%d %H:%M:%S')
            if datetime.now() - saved_time < timedelta(days=1):
                # Initialize model with saved parameters
                model = ForexLSTM(
                    input_size=input_size,
                    hidden_size=metadata['hidden_size'],
                    num_layers=metadata['num_layers']
                )
                # Load state with weights_only=True for security
                model.load_state_dict(torch.load(
                    model_path,
                    weights_only=True,  # Add this parameter
                    map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                ))
                print(f"Loaded saved model for {symbol}")
                return model
                
    except Exception as e:
        print(f"Error loading model: {e}")
    
    return None












# def implement_progressive_learning(model, mistake_patterns):
#     """Adjust model based on identified mistake patterns"""
#     for pattern, frequency in mistake_patterns.items():
#         if frequency > threshold:
#             # Add specific training data for this pattern
#             additional_data = generate_training_data_for_pattern(pattern)
#             model = train_on_specific_pattern(model, additional_data)
#     return model



# def monitor_learning_progress(symbol):
#     """Track if the model is actually improving"""
#     history_file = f'learning_history/{symbol}_progress.json'
#     current_metrics = analyze_prediction_accuracy(symbol)
    
#     with open(history_file, 'r+') as f:
#         history = json.load(f)
#         history.append({
#             'timestamp': datetime.now().isoformat(),
#             'metrics': current_metrics
#         })
        
#         # Analyze if learning is effective
#         recent_performance = history[-10:]  # Last 10 periods
#         is_improving = analyze_improvement_trend(recent_performance)
        
#         if not is_improving:
#             print("Learning not effective, adjusting strategy...")
#             adjust_learning_strategy(symbol)