# from fastapi import FastAPI, HTTPException
# import uvicorn

from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime 
from alpaca_trade_api import REST 
from datetime import datetime, timedelta
from sentiment import estimate_sentiment
import logging
import numpy as np
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import warnings

import os
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import alpaca
from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.stream import TradingStream
from alpaca.data.live.stock import StockDataStream

from alpaca.data.requests import (
    CorporateActionsRequest,
    StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest,
)
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetAssetsRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
)
from alpaca.trading.enums import (
    AssetExchange,
    AssetStatus,
    OrderClass,
    OrderSide,
    OrderType,
    QueryOrderStatus,
    TimeInForce,
)

import time



API_KEY = "PK88TRCFPEPZQ7P011WD" 
API_SECRET = "9o5RI3VZU5K6t9No3b1Vqmb3BlxAdffOrx6Wf7GX" 
BASE_URL = "https://paper-api.alpaca.markets/v2"

# API_KEY = "PKKR4TLMT2FA8JOP9JIE" 
# API_SECRET = "nk6M9TBAir01aWOk0i6DeaLYfPRayEscaTkB4hcD" 
# BASE_URL = "https://paper-api.alpaca.markets/v2"


ALPACA_CREDS = {
    "API_KEY":API_KEY, 
    "API_SECRET": API_SECRET, 
    "PAPER": True
}



class TradeBot(Strategy):
    def initialize(self, symbol:str="SPY", cash_at_risk:float=.05):
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None

        self.cash_at_risk = cash_at_risk
        self.trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
        self.stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)

        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        
        self.timeframe = "1DAY"

    def get_dates(self):
        print('========== GET DATES ============')
        try:
            today = datetime.now()
            three_days_prior = today - timedelta(days= 3)

            print(f'today: {today} - three_days_prior: {three_days_prior}')
            return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')
        
        except Exception as e:
            print(f'Error in get_dates: {e}')
            return None
        
    def sentiment_analysis(self):
        print('========== SENTIMENT ANALYSIS ============')
        try:
            today, three_days_prior = self.get_dates()

            print(f'today: {today} - three-day-prior: {three_days_prior}')

            news = self.api.get_news(
                symbol=self.symbol, 
                start=three_days_prior, 
                end=today
            ) 

            news = [ev.__dict__["_raw"]["headline"] for ev in news]


            if not news:
                print(f'No news found for sentiment analysis')
                return 0
            
            probability, sentiment = estimate_sentiment(news)

            return probability, sentiment
        
        except Exception as e:
            print(f'Error in sentiment_analysis: {e}')
            return 0, 0
        
    
    def position_size(self):
        print('========== POSITION SIZE ============')
        try:
            account = self.trading_client.get_account()
            print(f'account: {account.portfolio_value}')
            trade_value = float(account.portfolio_value) * self.cash_at_risk
            print(f'trade_value: {trade_value}')

            current_market_price = self.get_last_price()
            print(f'current_market_price: {current_market_price}')
            
            if not current_market_price:
                raise ValueError("Could not get current market price")
            
            min_trade_value = current_market_price * 0.01
            trade_value = max(min_trade_value, min(trade_value, float(account.portfolio_value)))
            print(f'Adjusted trade_value: {trade_value}')

            
            quantity = round(trade_value / current_market_price, 2)
            print(f'quantity: {quantity}')

            quantity = max(0.01, quantity)

            return quantity, trade_value, current_market_price
        
        except Exception as e:
            print(f'Error in position_size: {e}')
            return 0, 0, 0



    def get_last_price(self):
        print('========== GET LAST PRICE ============')
        try:
            start_date = self.get_datetime() - timedelta(days=360)
            end_date = self.get_datetime()
            print(f'start_date: {start_date.strftime("%Y-%m-%d")} - end_date: {end_date.strftime("%Y-%m-%d")}')
            request = StockBarsRequest(
                symbol_or_symbols=[self.symbol],
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=start_date.strftime('%Y-%m-%d'), 
                end=end_date.strftime('%Y-%m-%d')
                # start=self.get_datetime() - timedelta(days=5),  # Use strategy datetime
                # end=self.get_datetime() 
            )
            response = self.stock_client.get_stock_bars(request)
            
            if response and not response.df.empty:
                # print(response.df)
                last_price = float(response.df.iloc[-1].close)
                print(f"Last price for {self.symbol}: ${last_price:.2f}")
                return last_price
            return None
        except Exception as e:
            logging.error(f"Error getting last price: {e}")
            return None
        


    
    def calculate_bars(self, side, current_market_price, quantity):
        print('========== CALCULATE BARS ============')
        try:
            print(f'side: {side} - current_market_price: {current_market_price} - quantity: {quantity}')

            stop_loss_pct = 0.02
            take_profit_pct = 0.02

            if side == "buy":
                stop_loss_price = round(current_market_price * (1 - stop_loss_pct), 2)
                take_profit_price = round(current_market_price * (1 + take_profit_pct), 2)

            else:
                stop_loss_price = round(current_market_price * (1 + stop_loss_pct), 2)
                take_profit_price = round(current_market_price * (1 - take_profit_pct), 2)

            print(f'stop_loss_price: {stop_loss_price} - take_profit_price: {take_profit_price}')

            return stop_loss_price, take_profit_price
        
        except Exception as e:
            print(f'Error in calculate_bars: {e}')
            return None

    def submit_order(self, side, quantity, current_market_price):
        print('========== SUBMIT ORDER ============')
        try:
            stop_loss_price, take_profit_price = self.calculate_bars(side, current_market_price, quantity)
            notional_value = round(quantity * current_market_price, 2)
            if 0.01 <= quantity < 1:
                print("Fractional quantity proceeding")

                # notional_value = round(quantity * current_market_price, 2)

                print(f'current_market_price: {current_market_price}')
                print(f'quantity: {quantity}')
                print(f'side: {side}')  

            
                try:
                    # order_data = StopLimitOrderRequest(
                    #     symbol=self.symbol,
                    #     notional=notional_value,
                    #     side=OrderSide.SELL if side == "buy" else OrderSide.BUY,
                    #     time_in_force=TimeInForce.DAY,
                    #     type=OrderType.MARKET,
                    #     limit_price=take_profit_price,
                    #     stop_price=stop_loss_price
 
                    # )

                    order_data = MarketOrderRequest (
                        symbol=self.symbol,
                        notional=notional_value,
                        side=OrderSide.BUY if side != "buy" else OrderSide.SELL,
                        time_in_force=TimeInForce.DAY, # DAY
                        order_class=OrderClass.BRACKET,
                        take_profit=dict(
                            limit_price=stop_loss_price
                        ),
                        stop_loss=dict(
                            stop_price=take_profit_price,
                            limit_price=take_profit_price
                        ),
                        extended_hours=False
                    )

                    order = self.trading_client.submit_order(order_data)

                
                except Exception as e:
                    print(f'Error in submit_order: {e}')
                    return None
            
                
                
                
                print(f"Order submitted: {order} ")
                self.last_trade = side
                return order
            
            elif quantity >= 1:
                print("Non-fractional quantity proceeding")
                qty = round(quantity)
                print(f'rounded quantity: {qty}')
                order_data2 = MarketOrderRequest (
                    symbol=self.symbol,
                    qty=qty,
                    side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY, # DAY
                    order_class=OrderClass.BRACKET,
                    take_profit=dict(
                        limit_price=take_profit_price
                    ),
                    stop_loss=dict(
                        stop_price=stop_loss_price,
                        limit_price=stop_loss_price
                    ),
                    extended_hours=False
                )
                order = self.trading_client.submit_order(order_data2)
                print(f"Order submitted: {order}")
                return order
            
            else:
                print(f'Invalid quantity: {quantity}')
                return None
        
        except Exception as e:
            print(f'Error in submit_order: {e}')
            return None

    def on_trading_bar(self):
        print('========== ON TRADING BAR ============')
        try:
            probability, sentiment = self.sentiment_analysis()
            print(f'probability: {probability} - sentiment: {sentiment}')

            quantity, trade_value, current_market_price = self.position_size()
            print(f'quantity: {quantity} - trade_value: {trade_value} - current_market_price: {current_market_price}')

            current_portfolio_value = float(self.trading_client.get_account().portfolio_value)
            print(f'current_portfolio_value: {current_portfolio_value}')

            positions = self.trading_client.get_all_positions()
            print(f'positions: {positions}')

            position_exits = any(p.symbol == self.symbol for p in positions)
            print(f'position_exits: {position_exits}')

            if position_exits:
                position = next(p for p in positions if p.symbol == self.symbol)
                print(f'position: {position}')

                quantity = float(position.qty)
                entry_price = float(position.avg_entry_price)
                current_price = self.get_last_price()

                unrealized_pl = float(position.unrealized_pl)
                print(f'Unrealized P&L: {unrealized_pl:.2%}')


                if ((position.side == "long" and sentiment == "negative") or 
                    (position.side == "short" and sentiment == "positive") or 
                    unrealized_pl <= -0.02 or unrealized_pl >= 0.3):
                    print(f"Exiting position: {position.side} at {current_market_price:.2f}")
                    self.sell_all()
                    return
                
                else:
                    print("Position close failed - will retry next iteration")
                    return 
                
            else:
                if quantity > 1:
                    if probability > 0.7:  # High confidence level
                        if sentiment == "positive":
                            print(f"Selling {quantity} shares of {self.symbol} at {current_market_price:.2f} (Short Position)")
                            self.submit_order("sell", quantity, current_market_price)
                        elif sentiment == "negative":
                            print(f"Buying {quantity} shares of {self.symbol} at {current_market_price:.2f} (Closing Short Position)")
                            self.submit_order("buy", quantity, current_market_price)
                        elif sentiment == "neutral":
                            print(f"Selling {quantity} shares of {self.symbol} at {current_market_price:.2f} (Short Position)")
                            self.submit_order("sell", quantity, current_market_price)
                    return 
                else:
                    if probability > 0.7:  # High confidence level
                        if sentiment == "positive":
                            print(f"Buying {quantity} shares of {self.symbol} at {current_market_price:.2f} (Long Position)")
                            self.submit_order("buy", quantity, current_market_price)
                        elif sentiment == "negative":
                            print(f"Selling {quantity} shares of {self.symbol} at {current_market_price:.2f} (Closing Long Position)")
                            self.submit_order("sell", quantity, current_market_price)
                        elif sentiment == "neutral":
                            print(f"Buying {quantity} shares of {self.symbol} at {current_market_price:.2f} (Opportunistic Long)")
                            self.submit_order("buy", quantity, current_market_price)
                    return

            

        except Exception as e:
            print(f'Error in on_trading_bar: {e}')


    
    def sell_all(self):
        print('========== SELL ALL ============')
        try:
            self.trading_client.close_position(self.symbol)
            print(f"All positions closed for {self.symbol}")
        except Exception as e:
            print(f'Error in sell_all: {e}')

    
    def run(self, symbol='SPY'):
        try:
            self.initialize(symbol)

            while True:

                self.position_size()
                self.on_trading_bar()

                time.sleep(5)

        except Exception as e:
            print(f'Error in run: {e}')
            raise e
        

def backtest(symbol:str="SPY", cash_at_risk:float=.8):

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 11, 28)
    
    try:
        broker = Alpaca(ALPACA_CREDS)
        strategy = TradeBot(
            name='tradebot_strategy',
            broker=broker,
            parameters={
                "symbol": symbol,
                "cash_at_risk": cash_at_risk

            },
        )

        

        result = strategy.backtest(
            YahooDataBacktesting,
            start_date,
            end_date,
            parameters={
                "symbol": symbol,
                "cash_at_risk": cash_at_risk
            },
        )

        return result

    except Exception as e:
        print(f'Error in backtest: {e}')
        raise e
    



if __name__ == "__main__":
    MODE = "live"
    if MODE == "backtest":
        result = backtest(symbol="SPY", cash_at_risk=.2)
        print(f'result: {result}')
    else:
        broker = Alpaca(ALPACA_CREDS)
        bot = TradeBot(
            name='tradebot_strategy',
            broker=broker,
        )
        bot.run()

