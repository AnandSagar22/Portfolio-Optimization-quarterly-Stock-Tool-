import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization
import yfinance as yf

# on average there are 252 trading days in a year
NUM_TRADING_DAYS = 252
# we will generate random portfolios
NUM_PORTFOLIOS = 10000

# List of popular USA stock symbols
available_stocks = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'FB', 'NVDA', 'BRK-B', 'JPM', 'JNJ',
    'V', 'WMT', 'PG', 'UNH', 'DIS', 'HD', 'MA', 'PYPL', 'VZ', 'INTC'
]

#  REPLACE available_stocks FOR INDIAN STOCKS
''' available_stocks = [
     'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'KOTAKBANK.NS',
     'ITC.NS', 'BAJFINANCE.NS', 'LT.NS', 'AXISBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS',
     'HCLTECH.NS', 'TITAN.NS', 'SUNPHARMA.NS', 'WIPRO.NS', 'ULTRACEMCO.NS', 'MARUTI.NS'
] '''

# Display the available stock symbols
print("Available stock symbols:")
for i, stock in enumerate(available_stocks, 1):
    print(f"{i}. {stock}")

# Prompt the user to select five stock symbols from the list
selected_stocks = []
for i in range(5):
    while True:
        try:
            choice = int(input(f"Select stock symbol {i + 1} by number (1-{len(available_stocks)}): "))
            if 1 <= choice <= len(available_stocks):
                selected_stocks.append(available_stocks[choice - 1])
                break
            else:
                print(f"Please enter a number between 1 and {len(available_stocks)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# historical data - define START and END dates
# Change dates according to your needs
start_date = '2020-03-28'
end_date = '2025-03-28'


def download_data_yfinance(symbols):
    """Fetches historical data for the selected stocks and returns adjusted close prices (or close if adjusted unavailable)."""
    raw = yf.download(symbols, start=start_date, end=end_date, group_by='column')
    cols = raw.columns

    # Flatten column names to strings for searching
    if isinstance(cols, pd.MultiIndex):
        flat = [' '.join(col).strip() for col in cols]
    else:
        flat = list(cols)

    # First attempt: find adjusted close columns
    adj_mask = [('Adj' in name and 'Close' in name) for name in flat]
    if any(adj_mask):
        mask = adj_mask
        use_adj = True
    else:
        # fallback to regular close columns
        mask = [('Close' in name and not ('Adj' in name)) for name in flat]
        if not any(mask):
            raise ValueError(f"Couldn't locate any Close columns. Available columns: {flat}")
        use_adj = False
        print("Warning: 'Adj Close' not found, using 'Close' prices which are not adjusted for corporate actions.")

    data = raw.iloc[:, mask]

    # If MultiIndex, set columns to ticker symbols
    if isinstance(cols, pd.MultiIndex):
        data.columns = [col[1] for col, m in zip(cols, mask) if m]

    # Drop any stock column that contains missing data
    data = data.dropna(axis=1)
    return data


def show_data(data):
    data.plot(figsize=(10, 5))
    plt.title("Historical Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()


def calculate_return(data):
    log_return = np.log(data / data.shift(1))
    return log_return.dropna()


def show_statistics(returns):
    print("Mean Annual Return:")
    print(returns.mean() * NUM_TRADING_DAYS)
    print("\nCovariance Matrix:")
    print(returns.cov() * NUM_TRADING_DAYS)


def show_portfolios(returns, volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns / volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.title("Randomly Generated Portfolios")
    plt.show()


def generate_portfolios(returns):
    portfolio_means, portfolio_risks, portfolio_weights = [], [], []
    num_assets = returns.shape[1]
    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w))))
    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)


def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility, sharpe_ratio])


def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]


def optimize_portfolio(initial_weights, returns):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(returns.shape[1]))
    result = optimization.minimize(fun=min_function_sharpe,
                                   x0=initial_weights,
                                   args=(returns,),
                                   method='SLSQP',
                                   bounds=bounds,
                                   constraints=constraints)
    return result


def print_optimal_portfolio(optimum, returns):
    opt_weights = np.round(optimum['x'], 3)
    port_stats = statistics(opt_weights, returns)
    print("Optimal portfolio weights: ", opt_weights)
    print("Expected return, volatility, and Sharpe ratio: ", np.round(port_stats, 3))


def show_optimal_portfolio(opt, returns, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker='o', alpha=0.3)
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    opt_return, opt_volatility, _ = statistics(opt['x'], returns)
    plt.plot(opt_volatility, opt_return, 'g*', markersize=20, label='Optimal Portfolio')
    plt.title("Risk-Return Portfolio Optimization")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dataset = download_data_yfinance(selected_stocks)
    valid_stocks = dataset.columns.tolist()
    if len(valid_stocks) < len(selected_stocks):
        print("Warning: Some selected stocks did not have valid data and were removed.")
    else:
        print("All selected stocks downloaded successfully.")
    show_data(dataset)
    log_daily_returns = calculate_return(dataset)
    show_statistics(log_daily_returns)
    pweights, means, risks = generate_portfolios(log_daily_returns)
    show_portfolios(means, risks)
    optimum = optimize_portfolio(pweights[0], log_daily_returns)
    print_optimal_portfolio(optimum, log_daily_returns)
    show_optimal_portfolio(optimum, log_daily_returns, means, risks)
