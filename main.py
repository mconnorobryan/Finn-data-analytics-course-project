
# # main.py (in your project root directory)
"""
Main script for options pricing analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
from src.data.loader import get_stock_data, get_options_chain
from src.data.processor import calculate_historical_volatility, prepare_model_inputs

def main():
    # Step 1: Get data
    print("Getting stock data...")
    stock_data = get_stock_data("AAPL")
    
    print("\nGetting options data...")
    options_data = get_options_chain("AAPL")
    
    # Step 2: Process data
    print("\nCalculating historical volatility...")
    vol_data = calculate_historical_volatility(stock_data)
    
    print("\nPreparing data for modeling...")
    prepared_stock, prepared_options = prepare_model_inputs(vol_data, options_data)
    
    # Step 3: Print some results
    print("\nResults:")
    print(f"Current stock price: ${prepared_stock['Close'].iloc[-1]:.2f}")
    print(f"Current volatility: {prepared_stock['volatility'].dropna().iloc[-1]:.2%}")
    
    # Replace this section in main.py
    print("\nOptions data:")
    for expiry, data in prepared_options.items():
        print(f"\nExpiration: {expiry}, Number of options: {len(data)}")
    
    # Print some call options near the current price
    try:
        current_price = prepared_stock['Close'].iloc[-1]
        near_atm_calls = data[
            (data['option_type'] == 'call') & 
            (data['strike'] > current_price * 0.9) & 
            (data['strike'] < current_price * 1.1)
        ]
        
        if not near_atm_calls.empty:
            print("Near-the-money call options:")
            cols_to_show = ['strike', 'lastPrice', 'daysToExpiry']
            if 'impliedVolatility' in near_atm_calls.columns:
                cols_to_show.append('impliedVolatility')
            print(near_atm_calls[cols_to_show].head())
        else:
            print("No near-the-money call options found")
    except Exception as e:
        print(f"Error processing near-the-money options: {e}")

    if __name__ == "__main__":
        main()       
# %%
