import MetaTrader5 as mt5
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def calculate_win_rate(symbol):
    """Calculate win rate from MT5 trade history for a given symbol."""
    # Fetch trade history
    from_date = datetime.now() - timedelta(days=3)
    history_deals = mt5.history_deals_get(datetime(2024, 1, 1), datetime.now(), group=symbol)  # Adjust date range as needed
    if history_deals is None:
        print(f"No trade history found for {symbol}")
        return None

    total_trades = 0
    winning_trades = 0

    for deal in history_deals:
        total_trades += 1
        if deal.profit > 0:
            winning_trades += 1

    if total_trades == 0:
        print(f"No trades to evaluate for {symbol}.")
        return 0.0, 0, 0

    win_rate = (winning_trades / total_trades) * 100
    return win_rate, total_trades, winning_trades

def plot_win_rates(win_rates, symbols):
    """Plot the win rates for multiple symbols."""
    plt.figure(figsize=(10, 6))
    plt.bar(symbols, win_rates, color=['#4CAF50' if rate > 50 else '#F44336' for rate in win_rates])
    plt.axhline(y=50, color='gray', linestyle='--')  # Add a horizontal line at 50%
    plt.title('Win Rates for Trading Symbols')
    plt.xlabel('Symbols')
    plt.ylabel('Win Rate (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.show()

# Example usage
if __name__ == "__main__":
    if not mt5.initialize():
        print(f"Failed to initialize MT5: {mt5.last_error()}")
    else:
        symbols = ["AUDCAD", "CADCHF", "USDCHF", "EURUSD", "CADJPY", "USDCHF"]   # Replace with your array of symbols
        win_rates = []
        total_trades_list = []
        winning_trades_list = []

        for symbol in symbols:
            win_rate, total_trades, winning_trades = calculate_win_rate(symbol)
            if win_rate is not None:
                print(f"Win Rate for {symbol}: {win_rate:.2f}%")
                win_rates.append(win_rate)
                total_trades_list.append(total_trades)
                winning_trades_list.append(winning_trades)

        # Plot the win rates for all symbols
        plot_win_rates(win_rates, symbols)
        mt5.shutdown()