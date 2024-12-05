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
                
                # Check if the profit is greater than or equal to 10.10
                if current_profit >= 10.00: # 16.7
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
