# import MetaTrader5 as mt5
# import traceback

# from datetime import datetime, timedelta
# import pandas as pd
# import proba


# class Calendar:
#     def initialize_data(self):
#         self.ticker = proba.ticker
#         self.interval = proba.interval
#         self.from_date = proba.from_date
#         self.no_of_bars = 10 

#     def get_calender(self):
#         """Get the calendar for the current day"""
#         try:
#             if not mt5.initialize():
#                 print(f"Failed to initialize MT5: {mt5.last_error()}")
#                 return False
#             now = datetime.now()
#             start_time = now - timedelta(days=7)  # Last 7 days
#             end_time = now + timedelta(days=7) 

#             dfs = {}  # Local dictionary to store DataFrames
                
#             for tk in self.ticker:
#                 print(f"Fetching data for {tk}...")
#                 # Get historical data from MT5
#                 rates = mt5.copy_rates_from(tk, self.interval, self.from_date, self.no_of_bars)
                
#                 if rates is not None and len(rates) > 0:
#                         # Convert to DataFrame
#                         df = pd.DataFrame(rates)
#                         # Convert timestamp to datetime
#                         df['time'] = pd.to_datetime(df['time'], unit='s')
#                         df.name = tk
#                         dfs[tk] = df
#                         print(f"Retrieved {len(df)} rows for {tk}")
#                 else:
#                     print(f'No data found for {tk}')
                        
#             if not dfs:
#                 raise Exception("No data retrieved for any symbol")
                    
#             # Combine all dataframes
#             combined_df = pd.concat(dfs.values(), keys=dfs.keys(), axis=1)

#             print(f'combined_df: {combined_df}')
                
#             # Verify data size
#             print(f"\nData Summary:")
#             print(f"Total rows: {len(combined_df)}")
#             print(f"Columns per symbol: {len(combined_df.columns.levels[1])}")
                
#             return combined_df
#         except Exception as e:
#             print(f"Error getting calendar: {e}")
#             traceback.print_exc()
#             return None
        

    
#     def scrape_data(self):
#         import requests
#         from bs4 import BeautifulSoup

#         # Forex Factory Calendar URL
#         url = "https://www.forexfactory.com/calendar"

#         # Send a request to the Forex Factory calendar page
#         response = requests.get(url)
#         if response.status_code == 200:
#             soup = BeautifulSoup(response.text, 'html.parser')
            
#             # Find calendar table
#             calendar_table = soup.find('table', {'id': 'calendar-table'})
            
#             # Extract rows from the table
#             rows = calendar_table.find_all('tr', class_='calendar-row')
            
#             print("Forex Factory Economic Calendar:")
#             for row in rows:
#                 # Extract specific data points
#                 time = row.find('td', class_='time').get_text(strip=True)
#                 currency = row.find('td', class_='currency').get_text(strip=True)
#                 impact = row.find('td', class_='impact').get('title', '')  # Impact level tooltip
#                 event = row.find('td', class_='event').get_text(strip=True)
#                 actual = row.find('td', class_='actual').get_text(strip=True)
#                 forecast = row.find('td', class_='forecast').get_text(strip=True)
#                 previous = row.find('td', class_='previous').get_text(strip=True)
                
#                 # Print each event
#                 print(f"Time: {time}, Currency: {currency}, Event: {event}")
#                 print(f"Impact: {impact}, Actual: {actual}, Forecast: {forecast}, Previous: {previous}")
#                 print("-" * 50)
#         else:
#             print("Failed to retrieve Forex Factory calendar data.")

        
    

#     def run(self):
#         try:
#             self.initialize_data()
#             self.get_calender()
#         except Exception as e:
#             print(f"Error in run: {e}")
#             traceback.print_exc()
    


# if __name__ == "__main__":
#     try:
#         calendar = Calendar()
#         calendar.run()
#     except Exception as e:
#         print(f"Error in main: {e}")
#         traceback.print_exc()
#     finally:
#         mt5.shutdown()
#         print("MT5 connection closed")













import requests
from bs4 import BeautifulSoup
from datetime import datetime

class ForexNews:
    def __init__(self):
        self.url = "https://www.forexfactory.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
    def get_forex_news(self):
        try:
            print("Fetching Forex Factory news...")
            response = requests.get(f"{self.url}/calendar", headers=self.headers)
            
            if response.status_code != 200:
                print(f"Failed to fetch news. Status code: {response.status_code}")
                print(f"Response text: {response.text[:200]}")  # Print first 200 chars of response
                return []
                
            soup = BeautifulSoup(response.text, 'html.parser')
            calendar = soup.find('table', class_='calendar__table')
            
            if not calendar:
                print("No calendar found")
                return []
                
            news_items = []
            
            # Find all event rows
            for row in calendar.find_all('tr', class_='calendar__row calendar_row'):
                try:
                    # Extract time
                    time = row.find('td', class_='calendar__time')
                    time = time.text.strip() if time else "N/A"
                    
                    # Extract currency
                    currency = row.find('td', class_='calendar__currency')
                    currency = currency.text.strip() if currency else "N/A"
                    
                    # Extract impact
                    impact = row.find('td', class_='calendar__impact')
                    impact = impact.find('span')['class'][0] if impact and impact.find('span') else "low"
                    
                    # Extract event name
                    event = row.find('td', class_='calendar__event')
                    event = event.text.strip() if event else "N/A"
                    
                    # Only include if it's a relevant currency
                    if currency in ["EUR", "USD", "GBP", "JPY", "AUD", "CAD", "CHF"]:
                        news_items.append({
                            'time': time,
                            'currency': currency,
                            'event': event,
                            'impact': impact
                        })
                        
                except Exception as e:
                    print(f"Error parsing row: {e}")
                    continue
                    
            return news_items
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

    def print_news(self, news_items):
        if not news_items:
            print("\nNo news events found")
            return
            
        print("\n=== Forex Factory Calendar Events ===")
        print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        
        for item in news_items:
            print(f"\nTime: {item['time']}")
            print(f"Currency: {item['currency']}")
            print(f"Event: {item['event']}")
            print(f"Impact: {item['impact']}")
            print("-" * 50)

    def run(self):
        try:
            news = self.get_forex_news()
            self.print_news(news)
            return news
        except Exception as e:
            print(f"Error running news fetcher: {e}")
            return []

if __name__ == "__main__":
    news_fetcher = ForexNews()
    news_fetcher.run()