# New file: utils.py
import json
import os
from datetime import datetime, timedelta
import MetaTrader5 as mt5
# from update import calculate_advanced_metrics, get_adaptive_learning_rate
import torch
# from predictions import ForexLSTM
from learningMonitor import LearningMonitor
import numpy as np
from update import load_model
import traceback


learning_monitor = LearningMonitor()


def calculate_advanced_metrics(predictions):
    """Calculate advanced performance metrics"""
    try:
        # Calculate risk-adjusted returns
        returns = [p['actual_outcome']['profit'] for p in predictions if p['actual_outcome']]
        if not returns:
            return None
            
        returns = np.array(returns)
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        return {
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown)
        }
    except Exception as e:
        print(f"Error calculating advanced metrics: {e}")
        return None

def get_adaptive_learning_rate(performance):
    """Adjust learning rate based on performance"""
    try:
        base_lr = 0.001
        
        if performance['accuracy'] < 50:
            return base_lr * 2  # Increase learning rate when accuracy is poor
        elif performance['accuracy'] > 70:
            return base_lr * 0.5  # Decrease learning rate when performing well
            
        return base_lr
        
    except Exception as e:
        print(f"Error calculating adaptive learning rate: {e}")
        return 0.001  # Return default learning rate if error occurs
    



def save_prediction_results(symbol, prediction_data):
    """
    Save prediction results with trade tracking
    Args:
        prediction_data: Dictionary containing complete prediction and trade details
    """
    try:
        #print(f"\n In save_prediction_results: {prediction_data} \n with symbol: {symbol}")
        #symbol = prediction_data['symbol']
        
        # Create predictions directory if it doesn't exist
        if not os.path.exists('prediction_history'):
            os.makedirs('prediction_history')
            
        filename = f'prediction_history/{symbol}_predictions.json'
        
        # Load existing predictions
        existing_predictions = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_predictions = json.load(f)
                
        prediction_record = {
            'trade_id': prediction_data['trade_id'],
            'symbol': symbol,
            'entry_time': prediction_data['entry_time'],
            'entry_price': float(prediction_data['entry_price']),
            # 'current_price': float(prediction_data['current_price']),
            # 'predicted_price': float(prediction_data['predicted_price']),
            'predicted_direction': prediction_data['predicted_direction'],
            'predicted_change': float(prediction_data['predicted_change']),
            'stop_loss': float(prediction_data['stop_loss']),
            'take_profit': float(prediction_data['take_profit']),
            'position_size': float(prediction_data['position_size']),
            'status': prediction_data['status'],
            'mt5_ticket': prediction_data['mt5_ticket'],
            # 'movement': prediction_data['movement']
        }
        #print(f'prediction record in save_prediction_results: {prediction_data}')
        existing_predictions.append(prediction_record)
        
        # Keep only last 1000 predictions
        if len(existing_predictions) > 1000:
            existing_predictions = existing_predictions[-1000:]
            
        # Save updated predictions
        with open(filename, 'w') as f:
            json.dump(existing_predictions, f, indent=4)
            
        print(f"Prediction saved to {filename}")
        
    except Exception as e:
        print(f"Error saving prediction: {e}")



def update_prediction_outcomes(symbol):
    """Update outcomes for completed trades"""
    try:
        print(f"\nUpdating prediction outcomes for {symbol}")

        filename = f'prediction_history/{symbol}_predictions.json'
        if not os.path.exists(filename):
            print(f"No prediction history file found for {symbol}")
            return
            
        with open(filename, 'r') as f:
            predictions = json.load(f)
            
        # Get recent trade history
        from_date = datetime.now() - timedelta(days=2)
        history_deals = mt5.history_deals_get(from_date, datetime.now(), symbol)
        print(f'\n history deals in update prediction outcome : {history_deals} {"="*10}')
        
        if history_deals is None:
            print(f"No recent trade history found for {symbol}")
            return
            
        print(f"Found {len(history_deals)} recent trades to check")
        
        updated = False
        for deal in history_deals:
            # Find matching prediction by MT5 ticket
            for pred in predictions:
                if pred.get('mt5_ticket') == deal.ticket and pred['status'] != 'closed':
                    print(f"Updating outcome for trade ticket {deal.ticket}")
                    # Calculate actual outcome
                    entry_price = pred['entry_price']
                    exit_price = deal.price
                    actual_change = ((exit_price - entry_price) / entry_price) * 100
                    
                    pred['actual_outcome'] = {
                        'exit_time': datetime.fromtimestamp(deal.time).strftime('%Y-%m-%d %H:%M:%S'),
                        'exit_price': exit_price,
                        'actual_change': actual_change,
                        'movement': 'UP' if actual_change > 0 else 'DOWN',
                        'profit': deal.profit
                    }
                    pred['status'] = 'closed'
                    updated = True
                    print(f"Updated prediction outcome: profit={deal.profit}, movement={'UP' if actual_change > 0 else 'DOWN'}")

        if updated:
            # Save updated predictions
            with open(filename, 'w') as f:
                json.dump(predictions, f, indent=4)
            print(f"Saved updated predictions to {filename}")
            
            # Analyze updated accuracy
            analyze_prediction_accuracy(symbol)
            
    except Exception as e:
        print(f"Error updating prediction outcomes for {symbol}: {e}")
        print(f"Stack trace: {traceback.format_exc()}")





def analyze_prediction_accuracy(symbol):
    """
    Analyze prediction accuracy and adjust model parameters if needed
    """
    try:
        print(f'\n In analyze_prediction_accuracy: {symbol}')
        filename = f'prediction_history/{symbol}_predictions.json'
        if not os.path.exists(filename):
            return
            
        with open(filename, 'r') as f:
            predictions = json.load(f)
            
        # Filter predictions with actual outcomes
        completed_predictions = [p for p in predictions if p['actual_outcome'] is not None]
        
        if not completed_predictions:
            return
            
        # Calculate accuracy metrics
        total = len(completed_predictions)
        correct_direction = sum(1 for p in completed_predictions 
                              if p['predicted_movement'] == p['actual_outcome']['movement'])
        accuracy = (correct_direction / total) * 100
        
        # Calculate profit metrics
        total_profit = sum(p['actual_outcome']['profit'] for p in completed_predictions)
        win_rate = (sum(1 for p in completed_predictions if p['actual_outcome']['profit'] > 0) / total) * 100

        # Calculate advanced metrics
        advanced_metrics = calculate_advanced_metrics(completed_predictions)

        # Combine all metrics
        performance_metrics = {
            'accuracy': accuracy,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_trades': total
        }
        
        if advanced_metrics:
            performance_metrics.update(advanced_metrics)
        
        print(f"\nPrediction Analysis for {symbol} in analyze_prediction_accuracy:\n{'=' * 20}")
        print(f"Total Predictions: {total}")
        print(f"Correct Predictions: {correct_direction}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total Profit: {total_profit:.2f}")
        
        # If accuracy is below threshold and we have enough data, retrain model
        # if (accuracy < 55 or win_rate < 50) and total >= 50:
        #     print("Performance below threshold, initiating model retraining...")
            
        #     # Load existing model   
        #     input_size = 13  # Update this based on your feature set
        #     model = load_model(symbol, input_size)
            
        #     if model is not None:
        #         # Retrain model with feedback
        #         from predictions import retrain_model_with_feedback
        #         model = retrain_model_with_feedback(symbol, model, completed_predictions)
                
        #         # Reset prediction history after retraining
        #         predictions = predictions[-100:]  # Keep last 100 predictions
        #         with open(filename, 'w') as f:
        #             json.dump(predictions, f, indent=4)
        
        # return {
        #     'accuracy': accuracy,
        #     'win_rate': win_rate,
        #     'total_profit': total_profit
        # }


        # Determine if retraining is needed
        should_retrain = (
            accuracy < 55 or 
            win_rate < 50 or 
            (advanced_metrics and advanced_metrics['sharpe_ratio'] < 0.5)
        )
        
        if should_retrain and total >= 50:
            print(f"Performance below threshold, initiating model retraining...")
            
            # Get adaptive learning rate based on performance
            learning_rate = get_adaptive_learning_rate(performance_metrics)
            print(f"Using adaptive learning rate: {learning_rate}")
            
            # Load and retrain model
            input_size = 13  # Update based on your feature set
            model = load_model(symbol, input_size)
            
            from predictions import retrain_model_with_feedback

            if model is not None:
                model = retrain_model_with_feedback(
                    symbol=symbol,
                    model=model,
                    completed_predictions=completed_predictions,
                    learning_rate=learning_rate  # Pass the adaptive learning rate
                )
        

        # Track mistake patterns
        mistake_patterns = learning_monitor.track_mistake_patterns(completed_predictions, symbol)
        print(f"Mistake patterns: {mistake_patterns}")
        
        # Monitor learning progress
        is_improving = learning_monitor.monitor_learning_progress(symbol, performance_metrics)
        print(f"Is improving: {is_improving}")
        
        # Get recommendations
        recommendations = learning_monitor.get_learning_recommendations(symbol)
        for rec in recommendations:
            print(f"Recommendation: {rec}")
                
        return performance_metrics
                    
                     
    except Exception as e:
        print(f"Error analyzing prediction accuracy: {e}")



