# Portfolio-Optimization-Stock-Tool-
A lightweight Python tool demonstrating Modern Portfolio Theory: download historical prices from Yahoo Finance, compute log returns, generate random portfolios, visualize the efficient frontier, and find the Sharpe-optimal portfolio using constrained optimization.
Quarterly Portfolio Optimization — Simple Package

This repository contains a minimal, ready-to-run demo of Modern Portfolio Theory using historical price data from Yahoo Finance. The package is intentionally small so recruiters and peers can quickly run it locally or inside common code editors (VS Code, PyCharm, etc.).

Included files (simple)

README.md (this file)

requirements.txt (Python dependencies)

.gitignore

src/portfolio.py (main script — single-file runnable)

sample_output/example_report.txt (short output report you can include in the repo)

Quick description

A small Python script that:

Downloads historical adjusted close prices via yfinance.

Computes log daily returns and annualized mean/covariance.

Generates Monte Carlo portfolios and finds the Sharpe-optimal long-only portfolio using SLSQP.

Produces simple plots and prints a short example report.

Prerequisites

Python 3.9 or later

Internet connection (yfinance downloads data at runtime)

Run locally (terminal)

(Optional) Create and activate a virtual environment:

python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv/Scripts/Activate.ps1

Install dependencies:

pip install -r requirements.txt

Run the script from the repository root:

python src/portfolio.py

By default the script uses a non-interactive sample of 5 tickers (so it runs immediately). To run interactively, open src/portfolio.py and set INTERACTIVE = True near the top — the script will then prompt you to select tickers from the printed list.

Run inside Visual Studio Code or other editors

Open the project folder in VS Code.

If using VS Code, the Python extension will detect the virtual environment; select it from the bottom-right interpreter selector.

Open src/portfolio.py and press Run Python File or run the same terminal commands shown above inside VS Code's integrated terminal.

Files — short explanation

src/portfolio.py — contains the full script. It includes a toggle INTERACTIVE = False so it runs with a sample list by default. It also supports interactive selection if you prefer that flow.

requirements.txt — exact libraries to install.

sample_output/example_report.txt — a short example report (text) summarizing a sample run you can include for recruiters who prefer not to run the code.

.gitignore — excludes virtual environments, caches, and data.
