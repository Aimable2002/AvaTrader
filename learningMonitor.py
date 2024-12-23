import json
from datetime import datetime
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LearningMonitor:
    def __init__(self):
        """Initialize the learning monitor"""
        self.sessions = []
        try:
            # Create necessary directories if they don't exist
            Path("learning_history").mkdir(exist_ok=True)
            Path("mistake_patterns").mkdir(exist_ok=True)
            print("Learning monitor initialized. Created required directories.")
        except Exception as e:
            print(f"Error initializing learning monitor: {e}")


    def save_learning_history(self, filename='learning_history.json'):
        """Save the learning history to a JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.sessions, f, indent=4)
            print(f"Learning history saved to {filename}")
        except Exception as e:
            print(f"Error saving learning history: {e}")

    def load_learning_history(self, filename='learning_history.json'):
        """Load the learning history from a JSON file."""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.sessions = json.load(f)
            print(f"Learning history loaded from {filename}")
        else:
            print(f"No learning history file found at {filename}.")



    def record_training_session(self, session_data):
        """Record the details of a training session."""
        self.sessions.append(session_data)
        print(f"Recorded training session: {session_data}")


    def track_mistake_patterns(self, predictions, symbol):
        """
        Analyze patterns in incorrect predictions
        Args:
            predictions: List of prediction dictionaries with outcomes
            symbol: Trading symbol
        Returns:
            dict: Patterns of mistakes found
        """
        try:
            mistake_patterns = {}
            
            for pred in predictions:
                # Check if prediction was incorrect
                if pred.get('actual_outcome') and pred['predicted_direction'] != pred['actual_outcome']['movement']:
                    # Extract market conditions during mistake
                    entry_time = datetime.strptime(pred['timestamp'], '%Y-%m-%d %H:%M:%S')
                    conditions = {
                        'time_of_day': entry_time.hour,
                        'day_of_week': entry_time.weekday(),
                        'predicted_change': pred['predicted_change'],
                        'actual_change': pred['actual_outcome']['actual_change'],
                        'profit': pred['actual_outcome']['profit']
                    }
                    
                    # Create pattern key
                    pattern_key = (f"hour_{conditions['time_of_day']}_"
                                 f"day_{conditions['day_of_week']}_"
                                 f"change_{conditions['predicted_change']:.2f}")
                    
                    # Count pattern occurrence
                    mistake_patterns[pattern_key] = mistake_patterns.get(pattern_key, 0) + 1
            
            # Save mistake patterns to file
            pattern_file = f'mistake_patterns/{symbol}_patterns.json'
            with open(pattern_file, 'w') as f:
                json.dump(mistake_patterns, f, indent=4)
            
            # Print most common mistake patterns
            if mistake_patterns:
                print("\nMost Common Mistake Patterns:")
                sorted_patterns = sorted(mistake_patterns.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True)[:5]
                for pattern, count in sorted_patterns:
                    print(f"{pattern}: {count} occurrences")
            
            return mistake_patterns
            
        except Exception as e:
            print(f"Error tracking mistake patterns: {e}")
            return {}

    def monitor_learning_progress(self, symbol, current_metrics):
        """
        Track if the model is actually improving
        Args:
            symbol: Trading symbol
            current_metrics: Dictionary containing current performance metrics
        Returns:
            bool: True if model is improving, False otherwise
        """
        try:
            history_file = f'learning_history/{symbol}_progress.json'
            
            # Load existing history or create new
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            # Add current metrics with timestamp
            history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': current_metrics
            })
            
            # Keep only last 100 records
            if len(history) > 100:
                history = history[-100:]
            
            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=4)
            
            # Analyze improvement trend
            if len(history) >= 10:
                recent_metrics = history[-10:]
                
                # Calculate trends
                accuracy_trend = self._calculate_trend([m['metrics']['accuracy'] 
                                                      for m in recent_metrics])
                profit_trend = self._calculate_trend([m['metrics']['total_profit'] 
                                                    for m in recent_metrics])
                
                is_improving = accuracy_trend > 0 and profit_trend > 0
                
                print("\nLearning Progress Analysis:")
                print(f"Accuracy Trend: {'Improving' if accuracy_trend > 0 else 'Declining'}")
                print(f"Profit Trend: {'Improving' if profit_trend > 0 else 'Declining'}")
                
                return is_improving
            
            return True  # Default to True if not enough history
            
        except Exception as e:
            print(f"Error monitoring learning progress: {e}")
            return False

    def _calculate_trend(self, values):
        """Calculate the trend of a series of values"""
        try:
            if not values:
                return 0
            x = np.arange(len(values))
            y = np.array(values)
            z = np.polyfit(x, y, 1)
            return z[0]  # Return slope of trend line
        except Exception as e:
            print(f"Error calculating trend: {e}")
            return 0

    def get_learning_recommendations(self, symbol):
        """
        Generate recommendations for improving model performance
        Args:
            symbol: Trading symbol
        Returns:
            list: Recommendations for improvement
        """
        try:
            recommendations = []
            
            # Load mistake patterns
            pattern_file = f'mistake_patterns/{symbol}_patterns.json'
            if os.path.exists(pattern_file):
                with open(pattern_file, 'r') as f:
                    patterns = json.load(f)
                
                # Analyze patterns for recommendations
                if patterns:
                    most_common = max(patterns.items(), key=lambda x: x[1])
                    recommendations.append(
                        f"Most common mistake pattern: {most_common[0]} "
                        f"({most_common[1]} occurrences)")
            
            # Load learning history
            history_file = f'learning_history/{symbol}_progress.json'
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                if history:
                    recent = history[-10:]
                    avg_accuracy = np.mean([m['metrics']['accuracy'] for m in recent])
                    
                    if avg_accuracy < 55:
                        recommendations.append(
                            "Consider increasing training data or adjusting model parameters")
                    
                    if avg_accuracy < 45:
                        recommendations.append(
                            "Warning: Model performance is poor. Consider retraining with different features")
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]


    def plot_learning_progress(self):
        """Plot the learning progress based on recorded sessions."""
        if not self.sessions:
            print("No training sessions recorded.")
            return

        epochs = [session['epochs'] for session in self.sessions]
        losses = [session['final_loss'] for session in self.sessions]
        samples_trained = [session['samples_trained'] for session in self.sessions]

        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses, marker='o', label='Final Loss', color='blue')
        plt.title('Training Loss Over Sessions')
        plt.xlabel('Session Number')
        plt.ylabel('Loss')
        plt.xticks(epochs)
        plt.legend()


        # Plot samples trained
        plt.subplot(1, 2, 2)
        plt.plot(epochs, samples_trained, marker='o', label='Samples Trained', color='green')
        plt.title('Samples Trained Over Sessions')
        plt.xlabel('Session Number')
        plt.ylabel('Number of Samples')
        plt.xticks(epochs)
        plt.legend()

        plt.tight_layout()
        plt.show()

