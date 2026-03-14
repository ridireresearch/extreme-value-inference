import yfinance as yf
import pandas as pd
import numpy as np

def load_example_data(tickers=["SPY", "QQQ", "IWM", "BTC-USD"], start_date="2010-01-01", end_date=None):
    """
    Downloads historical adjusted close prices from Yahoo Finance.
    Returns a DataFrame where columns are tickers and rows are dates.
    """
    # yf.download returns a multi-index column if multiple tickers
    # or single index if one. We use adjusted close.
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close']
            else:
                prices = data['Close']
        else:
            if 'Adj Close' in data.columns:
                prices = data[['Adj Close']]
            else:
                prices = data[['Close']]
                
        # If single ticker, make sure the column is named by the ticker
        if len(tickers) == 1 and isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
            
        return prices.dropna(how='all')
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def load_csv_data(file_obj):
    """
    Loads pricing or return data from an uploaded Streamlit CSV file obj.
    Expects a 'Date' column and numeric columns for assets.
    """
    try:
        df = pd.read_csv(file_obj, index_col=0, parse_dates=True)
        return df.select_dtypes(include=[np.number])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return pd.DataFrame()
