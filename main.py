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
from plNews import get_news_sentiment
import json



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
        self.initialize()

        # Load saved model if exists
        try:
            for symbol in self.ticker:
                model_path = f'models/{symbol}_model.pth'
                metadata_path = f'models/{symbol}_metadata.json'
                print(f"Checking for model at: {model_path}")
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                # print(f"Loaded metadata for {symbol}: {metadata}")

                model = ForexLSTM(
                    input_size=metadata['input_size'],
                    hidden_size=metadata['hidden_size'],
                    num_layers=metadata['num_layers']
                )
                # print(f"Model architecture: {model}")

                if os.path.exists(model_path):
                    print(f"Model file found for {symbol}, attempting to load...")
                    self.model.load_state_dict(torch.load(model_path, weights_only=True))
                    print(f"Loaded saved model for {symbol} successfully")
                else:
                    print(f"No saved model found for {symbol}, using new model")
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
                # rates = mt5.copy_rates_from(tk, self.interval, self.from_date, self.no_of_bars)

                rates = mt5.copy_rates_from_pos(tk, self.interval, 0, self.no_of_bars)
                
                if rates is not None and len(rates) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(rates)
                    # Convert timestamp to datetime
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.name = tk
                    dfs[tk] = df
                    # print(f"Retrieved {len(df)} rows for {tk}")

                    # print(f"Last data point for {tk}: {df.iloc[-1]}")

                    last_rate = rates[-1]
                    last_time = datetime.fromtimestamp(last_rate['time'])
                    # print(f"\nLast rate: {last_rate}, Converted date: {last_time}")
                else:
                    print(f'No data found for {tk}')
                    
            if not dfs:
                raise Exception("No data retrieved for any symbol")
                
            # Combine all dataframes
            combined_df = pd.concat(dfs.values(), keys=dfs.keys(), axis=1)
            
            # Verify data size
            # print(f"\nData Summary:")
            # print(f"Total rows: {len(combined_df)}")
            # print(f"Columns per symbol: {len(combined_df.columns.levels[1])}")
            # print(f"Last data point: {combined_df.iloc[-1]}")
            
            return combined_df
            
        except Exception as e:
            print(f"Error getting price history: {e}")
            traceback.print_exc()
            return None
        
    
    def check_probability(self, df, symbols):
        """
        Check probability and generate trading signals
        Returns trading signals based on price movement predictions
        Buy signal when price is falling (expecting reversal up)
        Sell signal when price is rising (expecting reversal down)
        """

        try:
            print("\nAnalyzing market conditions...")
            # print(f"\n df in check_probability: {df.iloc[-1]}")
            predictions = predict_next_prices(df, symbols=self.ticker)

            # probability, sentiment = get_news_sentiment()

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

                    signal = "HOLD"
                    reason = "Default reason"

                    
                    # Generate trading signal (contrarian approach) and sentiment == "positive" or sentiment == "neutral"  and sentiment == "negative" or sentiment == "neutral"
                    # if probability > 0.7:
                    if price_movement == "UP" and change_percent > 0.1 :
                        signal = "BUY"  # BUY when price is expected to rise
                        reason = f"Price expected to rise by {change_percent:.2f}%, potential selling opportunity"
                    elif price_movement == "DOWN" and change_percent < -0.1:
                        signal = "SELL"   # SELL when price is expected to fall
                        reason = f"Price expected to fall by {change_percent:.2f}%, potential buying opportunity"
                    else:
                        signal = "HOLD"
                        reason = "Probability too small (less than 0.7)"
                        
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
                            'change_percent': change_percent,
                            'current_price': current_price,
                            'predicted_price': predicted_price,
                            'price_movement': price_movement
                        })
                        
                except Exception as e:
                    print(f"Error analyzing {symbol}: {str(e)}")
                    continue

            if potential_trades:
                min_change = 0.1  # Minimum 0.1% change required
                buy_trades = [trade for trade in potential_trades 
                            if trade['signal'] == "BUY" and trade['change_percent'] >= min_change]
                sell_trades = [trade for trade in potential_trades 
                            if trade['signal'] == "SELL" and trade['change_percent'] <= -min_change]
                
                # Find best buy and sell opportunities (keep original sign)
                best_buy = max(buy_trades, key=lambda x: x['change_percent']) if buy_trades else None  # Most negative change for BUY
                best_sell = min(sell_trades, key=lambda x: x['change_percent']) if sell_trades else None  # Most positive change for SELL
                
                print("\nBest Trade Analysis:")
                if best_buy:
                    print(f"Best BUY: {best_buy['symbol']} (Change: {best_buy['change_percent']:.2f}%)")  # Will show negative
                if best_sell:
                    print(f"Best SELL: {best_sell['symbol']} (Change: {best_sell['change_percent']:.2f}%)")  # Will show positive
                
                # Select both best buy and best sell trades
                selected_trades = []
                # Select both best buy and best sell trades
                selected_trades = []
                if best_buy:
                    selected_trades.append(best_buy)
                    print(f"Selected Best BUY: {best_buy['symbol']} (Change: {best_buy['change_percent']:.2f}%) selected_trades: {selected_trades}")
                if best_sell:
                    selected_trades.append(best_sell)
                    print(f"Selected Best SELL: {best_sell['symbol']} (Change: {best_sell['change_percent']:.2f}%) selected_trades: {selected_trades}")
                
                for best_trade in selected_trades:
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
                        time.sleep(1)
                    else:
                        print(f"\n Trade conditions not met: {position_details}")
                        print(f"Trade conditions not met: {position_details.get('reason', 'Unknown reason')} with should trade: {position_details.get('should_trade')}")
                

        except Exception as e:
            print(f"Error in check_probability: {e}")
            traceback.print_exc()

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
            signal_stop = position_details['signal']
            signal_limit = position_details['signal']
            trade_id = position_details['trade_id'] 

            print(f"Submitting order for {symbol} with signal {signal}")

            mt5_time = mt5.symbol_info_tick(symbol).time
            print(f"MT5 Time: {mt5_time}")
            if not mt5_time:
                print(f"Failed to fetch server time for {symbol}")
                return False  
            server_time = datetime.fromtimestamp(mt5_time, timezone.utc)
            print(f"Server Time above: {server_time}")
            expiration = server_time + timedelta(minutes=1440)
            print(f"Expiration Time above: {expiration}")

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position_details['size'],
                "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": float(position_details['entry_price']),
                "sl": float(position_details['stop_loss']),
                "tp": float(position_details['take_profit']),
                "deviation": self.deviation,
                "magic": 234000,
                "comment": f"LSTM {signal}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_time": mt5.ORDER_TIME_SPECIFIED,
                "type_filling": mt5.ORDER_FILLING_FOK,
                "expiration": int(expiration.timestamp())
            }

            
            point = mt5.symbol_info(symbol).point  
            min_distance = 5 * point  

            entry_distance = 1000 * point
            sl_distance = 500 * point      # 50 pips from entry
            tp_distance = 1000 * point  
            digits = mt5.symbol_info(symbol).digits

            # if signal_stop == "BUY":
            #     print(f"\n==== BUY STOP ====")
            #     # entry_price = mt5.symbol_info_tick(symbol).ask + min_distance
            #     # take_profit = entry_price + (30 * point)  
            #     # stop_loss = entry_price - (10 * point) 
            #     entry_price = round(mt5.symbol_info_tick(symbol).ask + entry_distance, digits)  
            #     stop_loss = round(entry_price - sl_distance, digits)           # 50 pips above entry
            #     take_profit = round(entry_price + tp_distance, digits)  
            # elif signal_stop == "SELL":
            #     print(f"\n==== SELL STOP ====")
            #     # entry_price = mt5.symbol_info_tick(symbol).bid - min_distance
            #     # take_profit = entry_price - (30 * point)  
            #     # stop_loss = entry_price + (10 * point) 
            #     entry_price = round(mt5.symbol_info_tick(symbol).bid - entry_distance, digits) 
            #     stop_loss = round(entry_price + sl_distance, digits)           # 50 pips below entry
            #     take_profit = round(entry_price - tp_distance, digits) 

            # request2 = {
            #     "action": mt5.TRADE_ACTION_PENDING,
            #     "symbol": symbol,
            #     "volume": position_details['size'],
            #     "type": mt5.ORDER_TYPE_BUY_STOP if signal_stop == "BUY" else mt5.ORDER_TYPE_SELL_STOP,
            #     "price": entry_price, 
            #     "sl": stop_loss,
            #     "tp": take_profit,
            #     "deviation": self.deviation,
            #     "magic": 234000,
            #     "comment": f"LSTM {signal_stop}",
            #     "type_time": mt5.ORDER_TIME_SPECIFIED,
            #     "type_filling": mt5.ORDER_FILLING_FOK,
            #     "expiration": int(expiration.timestamp())
            # }


            # symbol_info = mt5.symbol_info(symbol)
            # current_tick = mt5.symbol_info_tick(symbol)
            # point = symbol_info.point
            # digits = symbol_info.digits
            # stops_level = symbol_info.trade_stops_level
            
            # print(f"\nSymbol Info:")
            # print(f"Point: {point}")
            # print(f"Digits: {digits}")
            # print(f"Stops Level: {stops_level}")
            # print(f"Current Ask: {current_tick.ask}")
            # print(f"Current Bid: {current_tick.bid}")

            
            # min_distance = (stops_level + 10) * point

            # entry_distance = 1000 * point
            # sl_distance = 500 * point      # 50 pips from entry
            # tp_distance = 1000 * point  

            # if signal_limit == "BUY":
            #     # For SELL signal (BUY_LIMIT)
            #     print(f"\n==== BUY Limit ====")
            #     entry_price = round(current_tick.ask - entry_distance, digits)  
            #     stop_loss = round(entry_price - sl_distance, digits)           # 50 pips above entry
            #     take_profit = round(entry_price + tp_distance, digits)         # 100 pips below entry
            #     order_type = mt5.ORDER_TYPE_BUY_LIMIT
            # elif signal_limit == "SELL":
            #     # For BUY signal (SELL_LIMIT)
            #     print(f"\n==== SELL Limit ====")
            #     entry_price = round(current_tick.bid + entry_distance, digits) 
            #     stop_loss = round(entry_price + sl_distance, digits)           # 50 pips below entry
            #     take_profit = round(entry_price - tp_distance, digits)         # 100 pips above entry
            #     order_type = mt5.ORDER_TYPE_SELL_LIMIT

            # mt5_time = mt5.symbol_info_tick(symbol).time
            # server_time = datetime.fromtimestamp(mt5_time, timezone.utc)
            # expiration = server_time + timedelta(minutes=120)


            # request = {
            #     "action": mt5.TRADE_ACTION_PENDING,
            #     "symbol": symbol,
            #     "volume": position_details['size'],
            #     "type": order_type,
            #     "price": entry_price,
            #     "sl": stop_loss,
            #     "tp": take_profit,
            #     "deviation": self.deviation,
            #     "magic": 234000,
            #     "comment": f"LSTM {signal_limit}",
            #     "type_time": mt5.ORDER_TIME_SPECIFIED,
            #     "type_filling": mt5.ORDER_FILLING_FOK,
            #     "expiration": int(expiration.timestamp())
            # }


            print(f"Debug - Order request details:")

            for key, value in request.items():
                if key == "expiration":
                    # Convert expiration timestamp back to datetime for logging
                    expiration_datetime = datetime.fromtimestamp(value, timezone.utc)
                    print(f"expiration: {expiration_datetime} (timestamp: {value})")
                else:
                    print(f"{key}: {value}")

            # probability, sentiment = get_news_sentiment()

            # print(f"News Sentiment: {sentiment}")
            # print(f"News Sentiment Confidence: {probability:.2%}")

            result = mt5.order_send(request)

            # if probability > 0.998 and sentiment == "positive": # sell
            #     result = mt5.order_send(request2)
            # elif probability < 0.998 and sentiment == "negative": # buy
            #     result = mt5.order_send(request1)
            # else:
            #     result = mt5.order_send(request)

            if result is None:
                error = mt5.last_error()
                print("\nOrder failed!")
                print(f"MT5 error code: {error[0]}")
                print(f"MT5 error message: {error[1]}")
                return False

            time.sleep(2)
                
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
                time.sleep(2)
                mt5_ticket = result.order  # Use 'order' instead of 'ticket'
                print(f"Order placed successfully with ticket: {mt5_ticket}")

                # Update prediction record with MT5 ticket
                trade_prediction = {
                    'trade_id': trade_id,
                    'symbol': symbol,
                    'mt5_ticket': mt5_ticket,
                    'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'entry_price': position_details['entry_price'],
                    'predicted_direction': signal,
                    'predicted_change': position_details['predicted_change'],
                    # 'predicted_price': position_details['predicted_price'],
                    'stop_loss': position_details['stop_loss'],
                    'take_profit': position_details['take_profit'],
                    'position_size': position_details['size'],
                    'status': 'open',
                    # 'movement': position_details.get('movement', 'UNKNOWN') 
                }

                print(f"Trade Prediction: {trade_prediction}")
                
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
            print("\nMT5 End Run")

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
        print("\nMT5 connection closed")











# # Adjust entry price to respect minimum distance (20 points)
#             point = mt5.symbol_info(symbol).point  # Get point value
#             min_distance = 20 * point  # Minimum distance in price

#             if signal == "BUY":
#                 # For SELL_LIMIT, entry price must be ABOVE current Bid
#                 entry_price = mt5.symbol_info_tick(symbol).bid + min_distance
#                 take_profit = entry_price - (30 * point)  # TP below entry price
#                 stop_loss = entry_price + (10 * point)  # SL above entry price
#             else:
#                 # For BUY_LIMIT, entry price must be BELOW current Ask
#                 entry_price = mt5.symbol_info_tick(symbol).ask - min_distance
#                 take_profit = entry_price + (30 * point)  # TP above entry price
#                 stop_loss = entry_price - (10 * point)  # SL below entry price


#             request = {
#                 "action": mt5.TRADE_ACTION_PENDING,
#                 "symbol": symbol,
#                 "volume": position_details['size'],
#                 "type": mt5.ORDER_TYPE_SELL_LIMIT if signal == "BUY" else mt5.ORDER_TYPE_BUY_LIMIT,
#                 "price": entry_price, 
#                 "sl": float(position_details['stop_loss']),
#                 "tp": float(position_details['take_profit']),
#                 "deviation": self.deviation,
#                 "magic": 234000,
#                 "comment": f"LSTM {signal}",
#                 "type_time": mt5.ORDER_TIME_SPECIFIED,
#                 "type_filling": mt5.ORDER_FILLING_FOK,
#                 "expiration": int(expiration.timestamp())
#             }


