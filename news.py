# from alpaca_trade_api import REST
# from datetime import datetime, timedelta
# from lumibot.brokers import Alpaca
# from lumibot.strategies.strategy import Strategy
# from sentiment import estimate_sentiment

# API_KEY = "PK88TRCFPEPZQ7P011WD" 
# API_SECRET = "9o5RI3VZU5K6t9No3b1Vqmb3BlxAdffOrx6Wf7GX" 
# BASE_URL = "https://paper-api.alpaca.markets/v2"


# ALPACA_CREDS = {
#     "API_KEY":API_KEY, 
#     "API_SECRET": API_SECRET, 
#     "PAPER": True
# }
# import requests

# class botNews(Strategy):
#     def initialize(self, symbol:str="EUR"):
#         self.symbol = symbol
#         self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)


#     def get_dates(self):
#         three_days_prior = datetime.now() - timedelta(days=3)
#         today = datetime.now()
#         return three_days_prior.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')

#     def get_news(self):
#         try:
#             start, end = self.get_dates()
#             news = self.api.get_news(
#                 symbol=self.symbol, 
#                 start=start, 
#                 end=end
#             )
            
#             news = [ev.__dict__["_raw"]["headline"] for ev in news]

#             print(f'news: {news}')
#             probability, sentiment = estimate_sentiment(news)

#             print(f'probability: {probability}, sentiment: {sentiment}')

#             return probability, sentiment

#         except Exception as e:
#             print(f'Error: {e}')

    
#     def run(self, symbol:str="EUR"):
#         try:
#             self.initialize(symbol)
#             self.get_news()
#         except Exception as e:
#             print(f'Error: {e}')

# if __name__ == "__main__":
#     # news = News_alpaca()
#     # news.run()
#     broker = Alpaca(ALPACA_CREDS)
#     bot = botNews(
#         name='tradebot_strategy',
#         broker=broker,
#     )
#     bot.run()








from alpaca_trade_api.rest import REST
from datetime import datetime, timedelta
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

class AlpacaForex:
    def __init__(self):
        self.API_KEY = "PK88TRCFPEPZQ7P011WD" 
        self.API_SECRET = "9o5RI3VZU5K6t9No3b1Vqmb3BlxAdffOrx6Wf7GX" 
        self.BASE_URL = "https://paper-api.alpaca.markets/v2"
        
        # Initialize Alpaca API
        self.api = REST(
            key_id=self.API_KEY,
            secret_key=self.API_SECRET,
            base_url=self.BASE_URL
        )

    def get_forex_data(self, symbols=None, timeframe='15Min', start=None, limit=1000):
        if symbols is None:
            symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY']
            
        if start is None:
            start = datetime.now() - timedelta(days=30)

        try:
            all_data = {}
            
            for symbol in symbols:
                print(f"Fetching data for {symbol}...")
                
                # Use get_bars for forex data
                bars = self.api.get_bars(
                    symbol,
                    timeframe,
                    start=start.strftime('%Y-%m-%d'),
                    end=datetime.now().strftime('%Y-%m-%d'),
                    limit=limit,
                    adjustment='raw'
                ).df
                
                if not bars.empty:
                    all_data[symbol] = bars
                    print(f"Retrieved {len(bars)} bars for {symbol}")
                else:
                    print(f"No data found for {symbol}")

            return all_data

        except Exception as e:
            print(f"Error fetching forex data: {e}")
            return None

    def print_forex_data(self, data):
        if not data:
            print("No data to display")
            return

        for symbol, df in data.items():
            print(f"\n=== {symbol} Data ===")
            print(f"Latest {min(5, len(df))} bars:")
            print(df.tail())
            print("-" * 50)

    def run(self):
        try:
            start_date = datetime.now() - timedelta(days=30)
            
            # Forex pairs with correct format
            symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD']
            
            data = self.get_forex_data(
                symbols=symbols,
                timeframe='15Min',
                start=start_date,
                limit=1000
            )
            
            self.print_forex_data(data)
            return data
            
        except Exception as e:
            print(f"Error running forex data fetcher: {e}")
            return None

if __name__ == "__main__":
    forex = AlpacaForex()
    data = forex.run()