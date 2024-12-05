
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import pytz
import proba 
from predictions import predict_next_prices, ForexLSTM, prepare_forex_data
from calculate import Calculate
from utilis import update_prediction_outcomes, analyze_prediction_accuracy, save_prediction_results
from learningMonitor import LearningMonitor
import os
import torch
import traceback
import time 


class TradingBot:
    def __init__(self):
        """Initialize the TradingBot with default values"""
        self.ticker = None
        self.interval = None
        self.qty = None
        self.deviation = None
        self.no_of_bars = None
        self.from_date = None
        self.dfs = {}  
        self.calculator = Calculate() 
        self.model = ForexLSTM()
        self.learning_monitor = LearningMonitor()

        # Load saved model if exists
        try:
            model_path = 'models/forex_lstm.pth'
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
                print("Loaded saved model successfully")
            else:
                print("No saved model found, using new model")
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using new model instead")

    def initialize(self):
        """Initialize trading parameters from proba configuration"""
        self.ticker = proba.ticker
        self.interval = proba.interval
        self.qty = proba.qty
        self.deviation = proba.deviation
        self.no_of_bars = 1000  # Set to get 1000 bars for LSTM
        
        # Set timezone to UTC
        timezone = pytz.timezone("Etc/UTC")
        # Create 'datetime' object in UTC time zone to avoid the implementation of a local time zone
        self.from_date = datetime.now(timezone)
        print("Initialization completed")

    def check_account_info(self):
        """Check and print MT5 account information"""
        try:
            account_info = mt5.account_info()
            if account_info is not None:
                # Convert tuple to list to display account info
                account_info_dict = dict(list(account_info._asdict().items()))
                print("\nAccount Info:")
                for prop in account_info_dict:
                    print(f"  {prop} = {account_info_dict[prop]}")
            else:
                print("Failed to get account info")
        except Exception as e:
            print(f"Error checking account info: {e}")
            traceback.print_exc()

    def get_price_history(self):
        """
        Get historical price data for all symbols
        Returns:
            pd.DataFrame: Combined DataFrame with historical data for all symbols
        """
        try:
            dfs = {}  # Local dictionary to store DataFrames
            
            for tk in self.ticker:
                print(f"Fetching data for {tk}...")
                # Get historical data from MT5
                rates = mt5.copy_rates_from(tk, self.interval, self.from_date, self.no_of_bars)
                
                if rates is not None and len(rates) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(rates)
                    # Convert timestamp to datetime
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.name = tk
                    dfs[tk] = df
                    print(f"Retrieved {len(df)} rows for {tk}")
                else:
                    print(f'No data found for {tk}')
                    
            if not dfs:
                raise Exception("No data retrieved for any symbol")
                
            # Combine all dataframes
            combined_df = pd.concat(dfs.values(), keys=dfs.keys(), axis=1)
            
            # Verify data size
            print(f"\nData Summary:")
            print(f"Total rows: {len(combined_df)}")
            print(f"Columns per symbol: {len(combined_df.columns.levels[1])}")
            
            return combined_df
            
        except Exception as e:
            print(f"Error getting price history: {e}")
            return None
        
    
    def check_probability(self, df, symbols):
        """
        Check probability and generate trading signals
        Returns trading signals based on price movement predictions
        Buy signal when price is falling (expecting reversal up)
        Sell signal when price is rising (expecting reversal down)
        """
        print("\nAnalyzing market conditions...")
        predictions = predict_next_prices(df, symbols=self.ticker)

        potential_trades = []
        
        for symbol in symbols:
            try:
                if symbol not in predictions:
                    print(f"No predictions available for {symbol}")
                    continue
                    
                pred = predictions[symbol]
                current_price = pred['current_price']
                predicted_price = pred['predicted_price']
                change_percent = pred['change_percent']
                price_movement = pred['movement'] 


                # Print detailed analysis
                print("\nPrediction Results:")
                print("=" * 50)
                print(f"\nAnalysis for {symbol}:")
                print(f"Current Price: {current_price:.5f}")
                print(f"Predicted Next Price: {predicted_price:.5f}")
                print(f"Expected Change: {change_percent:.2f}%")
                print(f"Price Trend: {price_movement}")
                
                print("-" * 30)

                
                # Generate trading signal (contrarian approach)
                if price_movement == "UP" and abs(change_percent) > 0.1:
                    signal = "SELL"  # Sell when price is expected to rise
                    reason = f"Price expected to rise by {change_percent:.2f}%, potential selling opportunity"
                elif price_movement == "DOWN" and abs(change_percent) > 0.1:
                    signal = "BUY"   # Buy when price is expected to fall
                    reason = f"Price expected to fall by {abs(change_percent):.2f}%, potential buying opportunity"
                else:
                    signal = "HOLD"
                    reason = "Price movement too small (less than 0.1%)"
                
                print("\nDecision Taken:")
                print("=" * 50)
                print(f"Trading Signal: {signal}")
                print(f"Reason: {reason}")
                print("-" * 30)
                
                # If we have a trading signal, calculate position and submit order
                if signal in ["BUY", "SELL"]:
                    pred['signal'] = signal  # Add this line
                    predictions[symbol] = pred
                    potential_trades.append({
                        'symbol': symbol,
                        'signal': signal,
                        'change_percent': abs(change_percent),
                        'current_price': current_price,
                        'predicted_price': predicted_price,
                        'price_movement': price_movement
                    })
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {str(e)}")
                continue

        if potential_trades:
            best_trade = max(potential_trades, key=lambda x: x['change_percent'])
            print(f"\nSelected Best Trade:")
            print(f"Symbol: {best_trade['symbol']}")
            print(f"Signal: {best_trade['signal']}")
            print(f"Expected Change: {best_trade['change_percent']:.2f}%")
            
            # Create prediction info with signal for best trade
            pred_with_signal = {
                best_trade['symbol']: {
                    'current_price': best_trade['current_price'],
                    'predicted_price': best_trade['predicted_price'],
                    'change_percent': best_trade['change_percent'],
                    'movement': best_trade['price_movement'],
                    'signal': best_trade['signal']
                }
            }
            position_details = self.calculator.calculate_position(
                symbol=best_trade['symbol'],
                predictions=pred_with_signal,
                trading_bot=self 
            )

            if position_details['should_trade']:
                position_details['signal'] = best_trade['signal']
                self.submit_order(position_details)
            else:
                print(f"Trade conditions not met: {position_details.get('reason', 'Unknown reason')}")
                

    def submit_order(self, position_details):
        """
        Submit an order based on position details and trading signal
        """
        try:
            current_situation = mt5.symbol_info_tick(position_details['symbol'])
            if not position_details['should_trade']:
                print(f"No trade for {position_details.get('symbol', 'Unknown')}: {position_details.get('reason')}")
                return False

            symbol = position_details['symbol']
            signal = position_details['signal']
            trade_id = position_details['trade_id'] 

            print(f"Submitting order for {symbol} with signal {signal}")

            mt5_time = mt5.symbol_info_tick(symbol).time
            print(f"MT5 Time: {mt5_time}")
            if not mt5_time:
                print(f"Failed to fetch server time for {symbol}")
                return False  # Handle missing server time gracefully
            server_time = datetime.fromtimestamp(mt5_time, timezone.utc)
            print(f"Server Time above: {server_time}")
            expiration = server_time + timedelta(minutes=2)
            print(f"Expiration Time above: {expiration}")

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position_details['size'],
                "type": mt5.ORDER_TYPE_SELL if signal == "BUY" else mt5.ORDER_TYPE_BUY,
                "price": float(position_details['entry_price']),
                "sl": float(position_details['stop_loss']),
                "tp": float(position_details['take_profit']),
                "deviation": self.deviation,
                "magic": 234000,
                "comment": f"LSTM {signal}",
                # "type_time": mt5.ORDER_TIME_GTC,
                "type_time": mt5.ORDER_TIME_SPECIFIED,
                "type_filling": mt5.ORDER_FILLING_FOK,
                "expiration": int(expiration.timestamp())
            }

            print(f"Debug - Order request details:")
            for key, value in request.items():
                if key == "expiration":
                    # Convert expiration timestamp back to datetime for logging
                    expiration_datetime = datetime.fromtimestamp(value, timezone.utc)
                    print(f"expiration: {expiration_datetime} (timestamp: {value})")
                else:
                    print(f"{key}: {value}")

            # Submit the order
            print(f"\n ++++++++ Submitting order for {symbol} with signal {signal} ++++++++")
            while True:

                if request['type'] == mt5.ORDER_TYPE_SELL:
                    current_situation = mt5.symbol_info_tick(symbol)
                    if current_situation is None:
                        print(f"Failed to fetch current situation for {symbol}")
                        return False
                    elif current_situation.bid < request['price'] + 100.10:
                        print(f'current_situation.bid: {current_situation.bid}')
                        print(f'request["price"]: {request["price"]}')
                        result = mt5.order_send(request)
                        break
                    else:
                        print(f"Price is too high for {symbol}")
                        return False
                
                elif request['type'] == mt5.ORDER_TYPE_BUY:
                    current_situation = mt5.symbol_info_tick(symbol)
                    if current_situation is None:
                        print(f"Failed to fetch current situation for {symbol}")
                        return False
                    elif current_situation.ask > request['price'] - 100.10:
                        print(f'current_situation.ask: {current_situation.ask}')
                        print(f'request["price"]: {request["price"]}')
                        result = mt5.order_send(request)
                        break
                    else:
                        print(f"Price is too low for {symbol}")
                        return False
                time.sleep(1)
            if result is None:
                error = mt5.last_error()
                print("\nOrder failed!")
                print(f"MT5 error code: {error[0]}")
                print(f"MT5 error message: {error[1]}")
                return False
                
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"\nOrder failed with retcode: {result.retcode}")
                print(f"Comment: {result.comment}")
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"\nOrder Failed!")
                print(f"Symbol: {symbol}")
                print(f"Signal: {signal}")
                print(f"Error Code: {result.retcode}")
                print(f"Error Description: {result.comment}")
                return False
            else:
                # Update prediction record with MT5 ticket
                trade_prediction = {
                    'trade_id': trade_id,
                    'symbol': symbol,
                    'mt5_ticket': result.order,
                    'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'entry_price': position_details['entry_price'],
                    'predicted_direction': signal,
                    'predicted_change': position_details['predicted_change'],
                    'stop_loss': position_details['stop_loss'],
                    'take_profit': position_details['take_profit'],
                    'position_size': position_details['size'],
                    'status': 'open'
                }
                
                # Save trade prediction
                save_prediction_results(symbol, trade_prediction)

                
                print(f"\nOrder Successfully Placed!")
                print(f"Symbol: {symbol}")
                print(f"Type: {signal}")
                print(f"Size: {position_details['size']}")
                print(f"Entry Price: {position_details['entry_price']:.5f}")
                print(f"Stop Loss: {position_details['stop_loss']:.5f}")
                print(f"Take Profit: {position_details['take_profit']:.5f}")
                # print(f"Expected Change: {position_details['change_percent']:.2f}%")
                return True

        except Exception as e:
            print(f"Error submitting order: {str(e)}")
            traceback.print_exc()
            return False

    
    # def monitor_trades_and_predictions(self):
    #     """Monitor trades and update prediction outcomes"""
    #     try:
    #         # Get current positions
    #         positions = mt5.positions_get()

    #         if positions:
    #             for position in positions:
    #                 # Check if position is from our scalping bot
    #                 if position.magic == 234000:
    #                     # Get position open time
    #                     open_time = datetime.fromtimestamp(position.time)
    #                     current_time = datetime.now()
                        
    #                     # If position is open more than 5 minutes, close it
    #                     if (current_time - open_time).total_seconds() > 300:  # 300 seconds = 5 minutes
    #                         close_request = {
    #                             "action": mt5.TRADE_ACTION_CLOSE,
    #                             "position": position.ticket,
    #                             "magic": 234000,
    #                             "comment": "Scalping time limit reached"
    #                         }
                            
    #                         result = mt5.order_send(close_request)
    #                         if result.retcode != mt5.TRADE_RETCODE_DONE:
    #                             print(f"Error closing expired position: {result.comment}")
    #                         else:
    #                             print(f"Closed expired scalping position: {position.ticket}")
            
            
    #         # Get recent trade history
    #         from_date = datetime.now() - timedelta(days=1)
    #         history_deals = mt5.history_deals_get(from_date, datetime.now())
            
    #         if history_deals is not None:
    #             for deal in history_deals:
    #                 if deal.entry == mt5.DEAL_ENTRY_OUT:  # Position closed
    #                     # Update prediction outcomes
    #                     update_prediction_outcomes(deal.symbol)
    #                     # Analyze updated accuracy
    #                     analyze_prediction_accuracy(deal.symbol)

    #             # Calculate overall performance metrics
    #             total_profit = sum(deal.profit for deal in history_deals)
    #             win_count = sum(1 for deal in history_deals if deal.profit > 0)
    #             total_trades = len(history_deals)
                
    #             if total_trades > 0:
    #                 win_rate = (win_count / total_trades) * 100
    #                 print(f"\nOverall Performance:")
    #                 print(f"Total Trades: {total_trades}")
    #                 print(f"Win Rate: {win_rate:.2f}%")
    #                 print(f"Total Profit: {total_profit:.2f}")
                    
    #                 # If performance is poor, trigger model retraining
    #                 if win_rate < 50 and total_trades >= 20:
    #                     print("Poor overall performance detected, scheduling model retraining...")
    #                     for symbol in self.ticker:
    #                         analyze_prediction_accuracy(symbol)
            
    #         # Update outcomes for all symbols being traded
    #         for symbol in self.ticker:
    #             update_prediction_outcomes(symbol)
                
    #     except Exception as e:
    #         print(f"Error monitoring trades and predictions: {e}")



    def monitor_trades(self):
        """Monitor trades and close them if they reach a profit of 20.10"""
        try:
            positions = mt5.positions_get()
            
            if positions is None:
                return
                
            for position in positions:
                if position.magic == 234000:  # Check if it's our bot's trade
                    # Calculate the current profit
                    current_profit = position.profit
                    
                    print(f"\nPosition {position.ticket} profit check:")
                    print(f"Current Profit: {current_profit}")
                    
                    # Check if the profit is greater than or equal to 20.10
                    if current_profit >= 10.10:
                        print(f"Closing position {position.ticket} - profit target reached")
                        
                        # Prepare close request
                        close_request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "position": position.ticket,
                            "symbol": position.symbol,
                            "volume": position.volume,
                            "type": mt5.ORDER_TYPE_BUY if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL,
                            "price": mt5.symbol_info_tick(position.symbol).ask if position.type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(position.symbol).bid,
                            "deviation": 20,
                            "magic": 234000,
                            "comment": "Profit target reached",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        
                        # Send close request
                        result = mt5.order_send(close_request)
                        
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            print(f"Error closing position: {result.comment}")
                        else:
                            print(f"Position {position.ticket} closed successfully")
                            
                            # Optionally, send a new order after closing
                            # self.submit_order(self.calculate_position(position.symbol, self.predictions))
                            
        except Exception as e:
            print(f"Error in monitor_trades: {e}")
            traceback.print_exc()




    def run(self):
        """Main execution method"""
        try:
            # Initialize MT5
            if not mt5.initialize(path="C:\\Program Files\\Ava Trade MT5 Terminal\\terminal64.exe"):
                print(f"Failed to initialize MT5: {mt5.last_error()}")
                return
            
            print("MT5 initialized successfully")
            
            # Initialize bot parameters
            self.initialize()
            
            # Check account information
            self.check_account_info()
            
            # Get historical data
            print("\nFetching historical data...")
            df = self.get_price_history()

            # self.monitor_trades_and_predictions()
            # while True:
            #     try:
            #         self.monitor_trades()
                
            
            if df is None:
                print("Error: Failed to get historical data")
                return
                    
            if len(df) < 1000:
                print(f"Warning: Not enough historical data. Got {len(df)} rows, need 1000")
                return
                    
            self.check_probability(df, symbols=self.ticker)

            
            
        except Exception as e:
            print(f"An error occurred in main execution: {e}")
        finally:
            # Always shut down MT5 connection
            # mt5.shutdown()
            print("\nMT5 connection closed")

if __name__ == '__main__':
    # Create and run the trading bot
    print("Starting Trading Bot...")
    
    try:
        # Trading Cycle
        bot = TradingBot()
        bot.run()

        
        # Learning Cycle
        from retrain import RetrainBot
        retrain_bot = RetrainBot()
        retrain_bot.learning_cycle()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        mt5.shutdown()
