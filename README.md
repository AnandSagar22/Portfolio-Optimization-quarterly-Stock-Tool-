# Stock-Portfolio-Optimization-Tool
A Python tool demonstrating Modern Portfolio Theory: download historical prices from Yahoo Finance, compute log returns, generate random portfolios, visualize the efficient frontier, and find the Sharpe-optimal portfolio using constrained optimization.


---

## Included files
- `README.md` 
- `requirements.txt` (Python dependencies)    
- `src/stock_portfolio_optimization.py` (main script — single-file runnable)  
- `sample_output.txt`

---

## Quick description
A small Python script that:

- Downloads historical adjusted close prices via `yfinance`.  
- Computes log daily returns and annualized mean / covariance.  
- Generates Monte Carlo portfolios and finds the Sharpe-optimal long-only portfolio using SLSQP.  
- Produces simple plots and prints a short example report.

---

## Prerequisites
- Python 3.9 or later  
- Internet connection (yfinance downloads data at runtime)

---

## Run inside Visual Studio Code or other editors

Open the project folder in VS Code (or your editor).

If using VS Code, select the virtual environment interpreter in the bottom-right.

Open src/portfolio.py and press Run Python File, or run the same commands shown above in the integrated terminal.

---

## Files — short explanation

src/stock_portfolio_optimization.py — main script.

requirements.txt — libraries to install (numpy, pandas, matplotlib, scipy, yfinance).

sample_output.txt — short example for quick viewing.

