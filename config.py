# config.py
"""
Configuration settings for the Options Pricing and Greeks Analysis project.
"""

# Data settings
DEFAULT_TICKER = "AAPL"  # Default stock to analyze
DEFAULT_START_DATE = "2024-04-01"  # Start date for historical data
DEFAULT_END_DATE = "2025-04-30"  # End date for historical data
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"

# Model parameters
RISK_FREE_RATE = 0.05  # Default risk-free interest rate (5%)
TRADING_DAYS_PER_YEAR = 252  # Number of trading days in a year
DEFAULT_OPTION_TYPE = "call"  # Default option type

# Visualization settings
DEFAULT_CHART_WIDTH = 10
DEFAULT_CHART_HEIGHT = 6
DEFAULT_DPI = 100



# src/data/loader.py
"""
Functions to load stock price and options data from various sources.
"""

import os
import pandas as pd
import yfinance as yf
import datetime as dt
from typing import List, Optional, Dict, Any, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def get_stock_data(ticker: str = config.DEFAULT_TICKER,
                  start_date: str = config.DEFAULT_START_DATE,
                  end_date: str = config.DEFAULT_END_DATE) -> pd.DataFrame:
    """
    Fetch historical stock price data using yfinance.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame containing historical price data
    """
    try:
        # Fetch data
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        # Save raw data
        os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
        stock_data.to_csv(f"{config.RAW_DATA_DIR}/{ticker}_stock_data.csv")
        
        print(f"Successfully downloaded stock data for {ticker}")
        return stock_data
    
    except Exception as e:
        print(f"Error downloading stock data: {e}")
        return pd.DataFrame()


def get_options_chain(ticker: str = config.DEFAULT_TICKER) -> Dict[str, pd.DataFrame]:
    """
    Fetch current options chain data for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with expiration dates as keys and DataFrames of options data as values
    """
    try:
        # Get ticker object
        stock = yf.Ticker(ticker)
        
        # Get expiration dates
        expirations = stock.options
        
        if not expirations:
            print(f"No options data available for {ticker}")
            return {}
        
        # Get options data for each expiration
        options_data = {}
        for expiry in expirations:
            # Get both calls and puts
            calls = stock.option_chain(expiry).calls
            puts = stock.option_chain(expiry).puts
            
            # Add identifier column
            calls['option_type'] = 'call'
            puts['option_type'] = 'put'
            
            # Combine into single DataFrame for this expiration
            options_data[expiry] = pd.concat([calls, puts])
            
            # Save raw data
            os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
            options_data[expiry].to_csv(f"{config.RAW_DATA_DIR}/{ticker}_options_{expiry}.csv")
        
        print(f"Successfully downloaded options data for {ticker}")
        return options_data
    
    except Exception as e:
        print(f"Error downloading options data: {e}")
        return {}


def get_risk_free_rate() -> float:
    """
    Fetch current risk-free rate (placeholder for now).
    In a real implementation, this could pull data from a financial API.
    
    Returns:
        Current risk-free interest rate as a decimal
    """
    # For now, just return the default from config
    # In a real implementation, this would fetch the current Treasury yield
    return config.RISK_FREE_RATE


if __name__ == "__main__":
    # Test the functions
    stock_data = get_stock_data()
    print(stock_data.head())
    
    options_data = get_options_chain()
    for expiry, data in options_data.items():
        print(f"\nExpiration: {expiry}")
        print(data.head())


# config.py
"""
Configuration settings for the Options Pricing and Greeks Analysis project.
"""

# Data settings
DEFAULT_TICKER = "AAPL"  # Default stock to analyze
DEFAULT_START_DATE = "2024-01-01"  # Start date for historical data
DEFAULT_END_DATE = "2025-04-30"  # End date for historical data

# Model parameters
RISK_FREE_RATE = 0.05  # Default risk-free interest rate (5%)
TRADING_DAYS_PER_YEAR = 252  # Number of trading days in a year


# src/data/processor.py
"""
Functions for cleaning and preprocessing stock and options data.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import datetime as dt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def calculate_historical_volatility(stock_data: pd.DataFrame, 
                                   window: int = 30) -> pd.DataFrame:
    """
    Calculate rolling historical volatility from stock price data.
    
    Args:
        stock_data: DataFrame with stock prices (must have 'Close' column)
        window: Rolling window size in trading days
        
    Returns:
        DataFrame with added 'returns' and 'volatility' columns
    """
    # Make a copy to avoid modifying the original
    data = stock_data.copy()
    
    # Calculate daily returns
    data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Calculate rolling volatility (annualized)
    data['volatility'] = data['returns'].rolling(window=window).std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    
    return data


def clean_options_data(options_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Clean and preprocess options data.
    
    Args:
        options_data: Dictionary of options DataFrames by expiration date
        
    Returns:
        Dictionary of cleaned options DataFrames
    """
    cleaned_data = {}
    
    for expiry, data in options_data.items():
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Convert date columns to datetime
        if 'lastTradeDate' in df.columns:
            df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate'])
        
        # Calculate days to expiration
        expiry_date = dt.datetime.strptime(expiry, '%Y-%m-%d')
        df['daysToExpiry'] = (expiry_date - dt.datetime.now()).days
        
        # Handle missing or infinite values in key columns
        for col in ['impliedVolatility', 'volume', 'openInterest', 'lastPrice']:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(0)
        
        # Save processed data
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        df.to_csv(f"{config.PROCESSED_DATA_DIR}/processed_options_{expiry}.csv")
        
        cleaned_data[expiry] = df
    
    return cleaned_data


def prepare_model_inputs(stock_data: pd.DataFrame, 
                        options_data: Dict[str, pd.DataFrame],
                        risk_free_rate: float = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Prepare data for options pricing models.
    
    Args:
        stock_data: DataFrame with processed stock data
        options_data: Dictionary of processed options DataFrames
        risk_free_rate: Current risk-free rate (if None, use config default)
        
    Returns:
        Tuple of (prepared stock data, prepared options data)
    """
    # Process stock data
    prepared_stock = calculate_historical_volatility(stock_data)
    current_price = prepared_stock['Close'].iloc[-1]
    current_vol = prepared_stock['volatility'].dropna().iloc[-1]
    
    # Use provided risk-free rate or default
    if risk_free_rate is None:
        risk_free_rate = config.RISK_FREE_RATE
    
    # Process options data
    prepared_options = {}
    for expiry, data in options_data.items():
        df = data.copy()
        
        # Add current stock price and volatility
        df['underlying_price'] = current_price
        df['historical_volatility'] = current_vol
        df['risk_free_rate'] = risk_free_rate
        
        # Time to expiry in years
        df['T'] = df['daysToExpiry'] / 365
        
        prepared_options[expiry] = df
    
    return prepared_stock, prepared_options


if __name__ == "__main__":
    # For testing, you could implement a quick test here 
    from src.data.loader import get_stock_data, get_options_chain
    
    # Test with some data
    stock_data = get_stock_data()
    vol_data = calculate_historical_volatility(stock_data)
    print("Historical volatility calculated:")
    print(vol_data[['Close', 'returns', 'volatility']].tail())
    
    options_data = get_options_chain()
    cleaned_options = clean_options_data(options_data)
    for expiry, data in cleaned_options.items():
        print(f"\nCleaned options for {expiry}:")
        print(data.head())
# %%
