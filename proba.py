import MetaTrader5 as mt5
from datetime import datetime
import os

# MT5 credentials from environment variables
login = os.getenv('MT5_LOGIN')
password = os.getenv('MT5_PASSWORD')
server = os.getenv('MT5_SERVER')
path = os.getenv('MT5_PATH')

# Trading parameters
ticker = ["EURJPY", "EURGBP", "AUDUSD", "USDCHF","GBPJPY"]  
# ticker = ["EURUSD", "EURGBP", "EURCAD","GBPJPY", "EURJPY", "EURAUD", "EURCHF", "AUDUSD", "GBPUSD", "USDCHF", "USDJPY"]
interval = mt5.TIMEFRAME_M5
from_date = datetime.now() 
no_of_bars = 1000
qty = 0.01
deviation = 2

# These will be initialized after MT5 connection
point = None
price = None
stops_level = None
filling_modes = None
sl = None  # Add stop loss
tp = None  # Add take profit


# Risk management parameters
risk_per_trade = 0.02  # 2% risk per trade
min_change_threshold = 0.1  # Minimum price change threshold
sl_percentage = 0.003  # 0.3% stop loss
tp_percentage = 0.006  # 0.6% take profit