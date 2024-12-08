# import MetaTrader5 as mt5
# import traceback
# import time


# def initialize_mt5():
#     """Initialize MetaTrader 5 connection"""
#     if not mt5.initialize():
#         print(f"Failed to initialize MT5: {mt5.last_error()}")
#         return False
#     return True


# def monitor_trades():
#     """Monitor trades and close them if they reach a profit of 10.10"""
#     try:
#         positions = mt5.positions_get()
        
#         if positions is None:
#             return
            
#         for position in positions:
#             if position.magic == 234000:  # Check if it's our bot's trade
#                 # Calculate the current profit
#                 current_profit = position.profit
                
#                 print(f"\nPosition {position.ticket} profit check:")
#                 print(f"Current Profit: {current_profit}")

#                 if current_profit >= 2.20:
#                     print(f'Profit > 2.20, sending pending order for position: {position.ticket} and current profit: {current_profit}')
#                     send_pending_order(position)
#                 # Check if the profit is greater than or equal to 10.10
#                 if current_profit >= 10.00 or current_profit <= 2.10: # 16.7
#                     print(f"Closing position {position.ticket} - profit target reached")
                    
#                     # Prepare close request
#                     close_request = {
#                         "action": mt5.TRADE_ACTION_DEAL,
#                         "position": position.ticket,
#                         "symbol": position.symbol,
#                         "volume": position.volume,
#                         "type": mt5.ORDER_TYPE_BUY if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL,
#                         "price": mt5.symbol_info_tick(position.symbol).ask if position.type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(position.symbol).bid,
#                         "deviation": 20,
#                         "magic": 234000,
#                         "comment": "Profit target reached",
#                         "type_time": mt5.ORDER_TIME_SPECIFIED,
#                         "type_filling": mt5.ORDER_FILLING_FOK,
#                     }
                    
#                     # Send close request
#                     result = mt5.order_send(close_request)
                    
#                     if result.retcode != mt5.TRADE_RETCODE_DONE:
#                         print(f"Error closing position: {result.comment}")
#                     else:
#                         print(f"Position {position.ticket} closed successfully")
                            
#     except Exception as e:
#         print(f"Error in monitor_trades: {e}")
#         traceback.print_exc()



import MetaTrader5 as mt5
import traceback
import time


def initialize_mt5():
    """Initialize MetaTrader 5 connection"""
    if not mt5.initialize():
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        return False
    return True


def monitor_trades():
    """Monitor trades and close them if they reach a profit of 10.10"""
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

                modify_position_levels(position, current_profit)

                if current_profit >= 10.10 or current_profit <= -20.00: # 16.7
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
                        "type_time": mt5.ORDER_TIME_SPECIFIED,
                        "type_filling": mt5.ORDER_FILLING_FOK,
                    }
                    
                    # Send close request
                    result = mt5.order_send(close_request)
                    
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        print(f"Error closing position: {result.comment}")
                    else:
                        print(f"Position {position.ticket} closed successfully")
                            
    except Exception as e:
        print(f"Error in monitor_trades: {e}")
        traceback.print_exc()



def modify_position_levels(position, current_profit):
    """
    Move stop loss to secure a profit of 2.00 when the trade reaches a profit of 2.20
    and implement trailing stop logic.
    """
    try:
        # Define thresholds
        BREAKEVEN_THRESHOLD = 2.20  # When to move stop loss to secure profit
        SECURE_PROFIT = 2.00        # Profit to secure at break-even
        TRAILING_THRESHOLD = 5.00   # When to start trailing stop
        TRAILING_STEP = 3.00        # Trail by 3.00 units

        point = mt5.symbol_info(position.symbol).point

        # Move to break-even point
        if current_profit >= BREAKEVEN_THRESHOLD and current_profit < TRAILING_THRESHOLD:
            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = position.price_open + SECURE_PROFIT * point
                print(f"Moving stop loss to secure profit of 2.00: {new_sl}")
            else:  # SELL position
                new_sl = position.price_open - SECURE_PROFIT * point
                print(f"Moving stop loss to secure profit of 2.00: {new_sl}")

            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "position": position.ticket,
                "symbol": position.symbol,
                "sl": new_sl,
                "tp": position.tp,  # Keep original take profit
                "magic": 234000
            }

            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Stop loss moved to secure profit of 2.00 for position {position.ticket}")
            else:
                print(f"Failed to modify stop loss: {result.comment}")

        # Implement trailing stop
        elif current_profit >= TRAILING_THRESHOLD:
            profit_above_threshold = current_profit - TRAILING_THRESHOLD
            steps = int(profit_above_threshold / TRAILING_STEP)

            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = position.price_open + (steps * TRAILING_STEP)
                if new_sl > position.sl:  # Only move stop loss up
                    print(f"Updating trailing stop for BUY position:")
                    print(f"Moving stop loss up to: {new_sl}")

                    request = {
                        "action": mt5.TRADE_ACTION_MODIFY,
                        "position": position.ticket,
                        "symbol": position.symbol,
                        "sl": new_sl,
                        "tp": position.tp,
                        "magic": 234000
                    }

                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"Successfully updated trailing stop for position {position.ticket}")
                    else:
                        print(f"Failed to update trailing stop: {result.comment}")

            else:  # SELL position
                new_sl = position.price_open - (steps * TRAILING_STEP)
                if new_sl < position.sl:  # Only move stop loss down
                    print(f"Updating trailing stop for SELL position:")
                    print(f"Moving stop loss down to: {new_sl}")

                    request = {
                        "action": mt5.TRADE_ACTION_MODIFY,
                        "position": position.ticket,
                        "symbol": position.symbol,
                        "sl": new_sl,
                        "tp": position.tp,
                        "magic": 234000
                    }

                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"Successfully updated trailing stop for position {position.ticket}")
                    else:
                        print(f"Failed to update trailing stop: {result.comment}")

    except Exception as e:
        print(f"Error in modify_position_levels: {e}")
        traceback.print_exc()




def send_market_order(position):
    """Convert pending order to market order when profit reaches target"""
    try:
        tick = mt5.symbol_info_tick(position.symbol)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,  # Changed to DEAL for immediate execution
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_BUY if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL,
            "price": tick.ask if position.type == mt5.ORDER_TYPE_SELL else tick.bid,
            "deviation": 20,
            "magic": 234000,
            "comment": "Market order from monitor",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Error sending market order: {result.comment}")
        else:
            print(f"Market order executed successfully: Ticket {result.order}")
            
    except Exception as e:
        print(f"Error in send_market_order: {e}")



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
