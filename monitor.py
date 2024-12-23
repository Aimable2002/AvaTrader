import MetaTrader5 as mt5
import traceback
import time
import json
import os
from datetime import datetime


def initialize_mt5():
    """Initialize MetaTrader 5 connection"""
    if not mt5.initialize(path="C:\\Program Files\\Moneta Markets MT5 Terminal\\terminal64.exe"):
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        return False
    # if not mt5.login(account=3053765, password="#219$Cch"):
    #     print(f'Failed to login : {mt5.last_error()}')
    #     return False
    return True




kept_profit = {}
max_profits = {}


# bybit_api_key = RT5PB4DQJFKmujziO3
# bybit_secret_key = Z998nWx35tWTfWM2jMRJqI5N4f2rNFTSGjOG



def monitor_trades():
    """Monitor trades and close them if they reach a profit of 10.10"""
    try:
        positions = mt5.positions_get()

        
        
        if positions is None:
            return
        
        global kept_profit, max_profits
        allowed_loss = 1.50

        escape_loss = 2.50

        for position in positions:
            if position.magic == 234000:
                symbol = position.symbol
                current_profit = position.profit
                modify_position_levels(position, position.profit)
                
                print(f"\nPosition {position.ticket} profit check:")
                print(f"Current Profit: {current_profit}")

                # Only load history if we don't have this ticket's data
                if position.ticket not in max_profits:
                    # Try to load existing history for this ticket
                    saved_max, saved_kept = load_profit_history(symbol)
                    if str(position.ticket) in saved_max:
                        max_profits[position.ticket] = saved_max[str(position.ticket)]
                        kept_profit[position.ticket] = saved_kept[str(position.ticket)]
                    else:
                        # Initialize new ticket with 0
                        max_profits[position.ticket] = 0
                        kept_profit[position.ticket] = 0

                # Calculate the current profit
                current_profit = position.profit
                
                print(f"\nPosition {position.ticket} profit check:")
                print(f"Current Profit: {current_profit}")

                if position.ticket not in max_profits:
                    max_profits[position.ticket] = 0  # Start tracking from 0
                    kept_profit[position.ticket] = 0  # Start tracking from 0
                
                if current_profit > max_profits[position.ticket]:  # If we reach a new high
                    max_profits[position.ticket] = current_profit
                    print(f"New max profit recorded: {max_profits[position.ticket]}")

                    # if max_profits[position.ticket] > 10:
                    if max_profits[position.ticket] > kept_profit[position.ticket]:  
                        kept_profit[position.ticket] = max_profits[position.ticket]
                        print(f"New peak profit recorded: {kept_profit[position.ticket]}")
                        save_profit_history(max_profits, kept_profit, symbol)  # Save after updating kept_profit
                        

                print(f"max_profits[position.ticket]: {max_profits[position.ticket]}")
                print(f'kept_profit: {kept_profit[position.ticket]}')
                
                if kept_profit[position.ticket] >= 20.10:
                    drawdown = kept_profit[position.ticket] - current_profit
                    print(f'drop down kept profit: {kept_profit[position.ticket] - current_profit}')
                    print(f"Current profit: {current_profit}")
                    print(f"Peak profit: {kept_profit[position.ticket]}, Current drawdown: {drawdown}")
                    
                    if drawdown >= allowed_loss:
                        print(f"Closing position due to drawdown: Peak was {kept_profit[position.ticket]}, dropped by {drawdown}")
                        
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
                            "comment": f"Drawdown protection",
                            "type_time": mt5.ORDER_TIME_SPECIFIED,
                            "type_filling": mt5.ORDER_FILLING_FOK or mt5.ORDER_FILLING_IOC,
                        }
                        
                        # Send close request
                        result = mt5.order_send(close_request)

                        if result is None:
                            error = mt5.last_error()
                            print(f"Error sending order: {error}")
                            continue
                        
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            print(f"Error closing position: {result.comment}")
                        else:
                            print(f"Position {position.ticket} closed successfully")
                            save_profit_history(max_profits, kept_profit, symbol)  # Save after closing position
                            cleanup_closed_positions(symbol, position.ticket)

                elif kept_profit[position.ticket] >= 10.00:
                    drawdown = kept_profit[position.ticket] - current_profit
                    print(f'drop down kept profit: {kept_profit[position.ticket] - current_profit}')
                    print(f"Current profit: {current_profit}")
                    print(f"Peak profit: {kept_profit[position.ticket]}, Current drawdown: {drawdown}")
                    
                    if drawdown >= escape_loss:
                        print(f"Closing position due to drawdown: Peak was {kept_profit[position.ticket]}, dropped by {drawdown}")
                        
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
                            "comment": f"Drawdown protection",
                            "type_time": mt5.ORDER_TIME_SPECIFIED,
                            "type_filling": mt5.ORDER_FILLING_FOK or mt5.ORDER_FILLING_IOC,
                        }
                        
                        # Send close request
                        result = mt5.order_send(close_request)

                        if result is None:
                            error = mt5.last_error()
                            print(f"Error sending order: {error}")
                            continue
                        
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            print(f"Error closing position: {result.comment}")
                        else:
                            print(f"Position {position.ticket} closed successfully")
                            save_profit_history(max_profits, kept_profit, symbol)  # Save after closing position
                            cleanup_closed_positions(symbol, position.ticket)
                

                elif kept_profit[position.ticket] <= -25.00:
                    drawdown = kept_profit[position.ticket] - current_profit
                    print(f'drop down kept profit: {kept_profit[position.ticket] - current_profit}')
                    print(f"Current profit: {current_profit}")
                    print(f"Peak profit: {kept_profit[position.ticket]}, Current drawdown: {drawdown}")
                    # negative_loss = -3.00
                    # if drawdown <= negative_loss:
                    #     print(f"Closing position due to drawdown: Peak was {kept_profit[position.ticket]}, dropped by {drawdown}")
                        
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
                            "comment": f"Drawdown protection",
                            "type_time": mt5.ORDER_TIME_SPECIFIED,
                            "type_filling": mt5.ORDER_FILLING_FOK or mt5.ORDER_FILLING_IOC,
                    }
                        
                    # Send close request
                    result = mt5.order_send(close_request)

                    if result is None:
                        error = mt5.last_error()
                        print(f"Error sending order: {error}")
                        continue
                        
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        print(f"Error closing position: {result.comment}")
                    else:
                        print(f"Position {position.ticket} closed successfully")
                        save_profit_history(max_profits, kept_profit, symbol)  # Save after closing position
                        cleanup_closed_positions(symbol, position.ticket)
                


                
                            
    except Exception as e:
        print(f"Error in monitor_trades: {e}")
        traceback.print_exc()




def modify_position_levels(position, current_profit):
    """
    Move stop loss to secure a profit of 2.00 when the trade reaches a profit of 2.20
    and implement trailing stop logic.
    """
    try:
        TRAILING_THRESHOLD = 5.00   # When to start trailing stop 
        SECURE_PROFIT = 10.00        # Profit to secure at break-even

        point = mt5.symbol_info(position.symbol).point
        print(f"Point: {point}")

        # Implement trailing stop
        if current_profit >= TRAILING_THRESHOLD:
            # Calculate new stop loss based on current profit
            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = position.price_current - (SECURE_PROFIT * point) * 5
                if new_sl > position.sl:  # Only move stop loss up
                    print(f"Updating trailing stop for BUY position:")
                    print(f"Moving stop loss up to: {new_sl}")

                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position.ticket,
                        "sl": new_sl,
                        "tp": position.tp,
                    }

                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"Successfully updated trailing stop for position {position.ticket}")
                    else:
                        print(f"Failed to update trailing stop: {result.comment}")

            else:  # SELL position
                new_sl = position.price_current + (SECURE_PROFIT * point) * 5
                if new_sl < position.sl:  # Only move stop loss down
                    print(f"Updating trailing stop for SELL position:")
                    print(f"Moving stop loss down to: {new_sl}")

                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position.ticket,
                        "sl": new_sl,
                        "tp": position.tp,
                    }

                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"Successfully updated trailing stop for position {position.ticket}")
                    else:
                        print(f"Failed to update trailing stop: {result.comment}")

    except Exception as e:
        print(f"Error in modify_position_levels: {e}")
        traceback.print_exc()









def save_profit_history(max_profits_data, kept_profit_data, symbol):
    """Save max_profits and kept_profit to file for specific symbol"""
    try:
        if not os.path.exists('profit_data'):
            os.makedirs('profit_data')
            
        data = {
            'max_profits': max_profits_data,
            'kept_profit': kept_profit_data,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        filename = f'profit_data/{symbol}_profit_history.json'
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Profit history saved for {symbol}")
        print(f"Saved max_profits: {max_profits_data}")
        print(f"Saved kept_profit: {kept_profit_data}")
            
    except Exception as e:
        print(f"Error saving profit history: {e}")
        traceback.print_exc()

def load_profit_history(symbol):
    """Load saved max_profits and kept_profit for specific symbol"""
    try:
        filename = f'profit_data/{symbol}_profit_history.json'
        if not os.path.exists(filename):
            print(f"No saved profit history found for {symbol}")
            return {}, {}
            
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # Convert string keys back to integers
        max_profits = {int(k): v for k, v in data['max_profits'].items()}
        kept_profit = {int(k): v for k, v in data['kept_profit'].items()}
        
        print(f"Loaded profit history for {symbol}")
        print(f"Loaded max_profits: {max_profits}")
        print(f"Loaded kept_profit: {kept_profit}")
        
        return max_profits, kept_profit
            
    except Exception as e:
        print(f"Error loading profit history: {e}")
        traceback.print_exc()
        return {}, {}

def cleanup_closed_positions(symbol, ticket):
    """Remove profit history for closed positions"""
    try:
        filename = f'profit_data/{symbol}_profit_history.json'
        if os.path.exists(filename):
            # Load existing data
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Remove closed position data
            if str(ticket) in data['max_profits']:
                del data['max_profits'][str(ticket)]
            if str(ticket) in data['kept_profit']:
                del data['kept_profit'][str(ticket)]
            
            # Save updated data
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
                
            print(f"Cleaned up profit history for closed position {ticket} on {symbol}")
            
    except Exception as e:
        print(f"Error cleaning up profit history: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    if not initialize_mt5():
        exit()
    
    try:
        while True:
            monitor_trades()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Monitoring stopped by user")
    finally:
        mt5.shutdown()
        print("MT5 connection closed")



