import MetaTrader5 as mt5
#from update import save_prediction_results, update_prediction_outcomes, analyze_prediction_accuracy
import json
from utilis import save_prediction_results, update_prediction_outcomes, analyze_prediction_accuracy, load_model
import os
from datetime import datetime
import numpy as np


class Calculate:
    def calculate_position(self, symbol, predictions, trading_bot=None):
        """
        Calculate position size and determine if we should enter a trade
        Args:
            symbol: Trading pair symbol
            predictions: Dictionary containing prediction data from LSTM
        Returns:
            dict: Position details including size, type, and entry price
        """
        try:
            performance = analyze_prediction_accuracy(symbol)
            if performance is None:
                # return {'should_trade': False, 'reason': 'Unable to analyze performance'}
                
                performance = {
                    'accuracy': 100,  # Start optimistic
                    'win_rate': 100,
                    'total_profit': 0,
                    'total_trades': 0
                }
           
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                raise Exception(f"Symbol {symbol} not found")
            
            current_tick = mt5.symbol_info_tick(symbol)
            if current_tick is None:
                raise Exception(f"Failed to get current price for {symbol}")


            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                raise Exception("Failed to get account info")

            # Get prediction details from LSTM
            pred = predictions[symbol]
            current_price = np.float64(pred['current_price']).item() 
            predicted_price = np.float64(pred['predicted_price']).item()
            predicted_change = np.float64(pred['change_percent']).item()
            signal = pred['signal']

            entry_price = current_price
            stop_loss = None
            take_profit = None


            # Get prediction accuracy before deciding on trade
            recent_accuracy = 0.55 
            filename = f'prediction_history/{symbol}_predictions.json'
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    historical_predictions = json.load(f)
                
            
            
            # Calculate position size based on predicted movement
            if abs(predicted_change) < 0.07:  # If predicted change is too small
                return {
                    'should_trade': False,
                    'reason': 'Predicted change too small'
                }
            # Calculate stop loss and take profit
            # Position sizing calculation (using 2% risk per trade)
            

            # Adjust position size based on performance
            if performance['total_trades'] > 0:  # Only adjust if we have trading history
                if performance['accuracy'] < 55:
                    position_size *= 0.5  # Reduce position size by 50%
                if performance['win_rate'] < 50:
                    position_size *= 0.5  # Further reduce if win rate is low
                if performance['total_profit'] < 0:
                    position_size *= 0.75  # Reduce if currently unprofitable
                    
                if performance['accuracy'] < 45:
                    return {'should_trade': False, 'reason': 'Accuracy too low'}


            # For scalping, we need a minimum predicted change to ensure potential profit
            MIN_CHANGE_THRESHOLD = 0.07  # 0.05% minimum expected change
            if abs(predicted_change) < MIN_CHANGE_THRESHOLD:
                return {
                    'should_trade': False,
                    'reason': f'Predicted change ({predicted_change:.3f}%) below minimum threshold ({MIN_CHANGE_THRESHOLD}%)'
                }

            # Check current spread
            current_spread = (symbol_info.ask - symbol_info.bid) / symbol_info.point
            point = symbol_info.point
                
            
            if signal != "BUY":
                entry_price = mt5.symbol_info_tick(symbol).bid
                print(f'bid: {entry_price}')
                stop_loss = entry_price * 0.997  # 0.3% stop loss 0.997 
                take_profit = entry_price * 1.006  # 0.6% take profit 1.006
            else:  # SELL
                entry_price = mt5.symbol_info_tick(symbol).ask
                print(f'ask: {entry_price}')
                stop_loss = entry_price * 1.003  # 0.3% stop loss 1.003
                take_profit = entry_price * 0.994 # 0.6% take profit 0.994 

            # min_stops_level = symbol_info.trade_stops_level
            # print(f'min_stops_level: {min_stops_level}')
            # point = symbol_info.point
            # print(f'point: {point}')
            # tick = mt5.symbol_info_tick(symbol)

            # if signal != "BUY":
            #     entry_price = tick.bid
            #     print(f'bid: {entry_price}')
            #     # For SELL orders: SL above entry, TP below entry
            #     stop_loss = entry_price + (min_stops_level * 2 * point)  # Double the minimum distance
            #     take_profit = entry_price - (min_stops_level * 4 * point)  # 2:1 reward:risk ratio
            #     order_type = mt5.ORDER_TYPE_SELL
            # else:  # BUY
            #     entry_price = tick.ask
            #     print(f'ask: {entry_price}')
            #     # For BUY orders: SL below entry, TP above entry
            #     stop_loss = entry_price - (min_stops_level * 2 * point)
            #     take_profit = entry_price + (min_stops_level * 4 * point)
            #     order_type = mt5.ORDER_TYPE_BUY

            
            account_balance = account_info.balance
            risk_amount = float(account_balance * 0.005)

            min_lot = symbol_info.volume_min
            lot_step = symbol_info.volume_step

            pip_value = float(symbol_info.trade_tick_value)

            stop_loss_pips = abs(entry_price - stop_loss) / float(symbol_info.point)
            position_size = risk_amount / (stop_loss_pips * pip_value)
            position_size = round(position_size / symbol_info.volume_step) * symbol_info.volume_step

            position_size = max(symbol_info.volume_min, 
                              min(position_size, symbol_info.volume_max))

            # max_allowed_size = float(account_balance * 0.01) / entry_price
            # max_allowed_size = round(max_allowed_size / lot_step) * lot_step

            # position_size = max(min_lot, min(position_size, max_allowed_size))

            if position_size < min_lot:
                return {
                    'should_trade': False,
                    'reason': f'Calculated position size ({position_size}) below minimum lot size ({min_lot})'
                }

            print(f"Debug - Position Size Calculation:")
            print(f"Risk Amount: {risk_amount}")
            print(f"Stop Loss Pips: {stop_loss_pips}")
            print(f"Position Size: {position_size}")
            # print(f"Min Stops Level: {min_stops_level} points")
            print(f"Entry Price: {entry_price}")
            print(f"Stop Loss: {stop_loss} ({abs(entry_price - stop_loss)/point} points)")
            print(f"Take Profit: {take_profit} ({abs(entry_price - take_profit)/point} points)")
            # print(f"Max Allowed Size: {max_allowed_size}")

            trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            trade_prediction = {
                'trade_id': trade_id,
                'symbol': symbol,
                'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': entry_price,
                'current_price': current_price,
                'predicted_direction': signal,
                'predicted_change': predicted_change,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size, 
                'status': 'pending',
                'mt5_ticket': None,
                'movement': pred['movement'],
                'strategy': 'scalping_5m',
                'spread_points': current_spread
            } 

            save_prediction_results(symbol, trade_prediction)
            update_prediction_outcomes(symbol)



            return {
                'should_trade': True,
                'symbol': symbol,
                'signal': signal,
                'type': 'sell' if signal == "SELL" else 'buy', 
                'size': position_size,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'predicted_change': predicted_change,
                'trade_id': trade_id,
                'spread': current_spread
            }

        except Exception as e:
            print(f"Error calculating position for {symbol}: {str(e)}")
            return {
                'should_trade': False,
                'reason': f'Error: {str(e)}'
            }
        

























    def submit_order(self, position_details, deviation=20):
        """
        Submit an order to MT5 based on position details
        Args:
            position_details: Dictionary containing position information
            deviation: Maximum price deviation
        """
        try:
            if not position_details['should_trade']:
                print(f"No trade for {position_details.get('symbol', 'Unknown')}: {position_details.get('reason')}")
                return

            symbol = position_details['symbol']
            order_type = mt5.ORDER_TYPE_BUY if position_details['type'] == 'buy' else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position_details['size'],
                "type": order_type,
                "price": position_details['entry_price'],
                "sl": position_details['stop_loss'],
                "tp": position_details['take_profit'],
                "deviation": deviation,
                "magic": 234000,  # Magic number for identifying bot trades
                "comment": "LSTM prediction trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # result = mt5.order_send(request)
            
            # if result.retcode != mt5.TRADE_RETCODE_DONE:
            #     print(f"Order failed: {result.comment}")
            #     return False
            # else:
            #     print(f"\nOrder successful: {symbol} {position_details['type'].upper()}")
            #     print(f"Size: {position_details['size']}")
            #     print(f"Entry: {position_details['entry_price']:.5f}")
            #     print(f"SL: {position_details['stop_loss']:.5f}")
            #     print(f"TP: {position_details['take_profit']:.5f}")
            #     print(f"Predicted Change: {position_details['predicted_change']:.2f}%")
            #     return True

        except Exception as e:
            print(f"Error submitting order: {str(e)}")
            return False
        

