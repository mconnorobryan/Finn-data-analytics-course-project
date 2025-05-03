# src/visualization/option_charts.py
"""
Functions for visualizing option prices and related data.

This module provides tools to create various charts for options analysis,
including price curves, implied volatility surfaces, and pricing comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from typing import Dict, List, Tuple, Any, Optional
import seaborn as sns
from datetime import datetime, timedelta

import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# For 3D plots
from mpl_toolkits.mplot3d import Axes3D


def plot_option_price_curve(strikes: List[float], 
                           call_prices: List[float], 
                           put_prices: List[float] = None,
                           underlying_price: float = None,
                           title: str = "Option Price Curve",
                           save_path: str = None) -> plt.Figure:
    """
    Plot option prices against strike prices.
    
    Args:
        strikes: List of strike prices
        call_prices: List of call option prices
        put_prices: List of put option prices (optional)
        underlying_price: Current stock price (to mark on the x-axis)
        title: Chart title
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot call prices
    ax.plot(strikes, call_prices, 'b-', marker='o', label='Call Prices')
    
    # Plot put prices if provided
    if put_prices is not None:
        ax.plot(strikes, put_prices, 'r-', marker='s', label='Put Prices')
    
    # Mark the underlying price if provided
    if underlying_price is not None:
        ax.axvline(x=underlying_price, color='k', linestyle='--', alpha=0.5, 
                  label=f'Underlying Price (${underlying_price:.2f})')
    
    # Set labels and title
    ax.set_xlabel('Strike Price ($)')
    ax.set_ylabel('Option Price ($)')
    ax.set_title(title)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_price_comparison(strikes: List[float],
                         theoretical_prices: List[float],
                         market_prices: List[float],
                         option_type: str = 'call',
                         title: str = None,
                         save_path: str = None) -> plt.Figure:
    """
    Plot comparison between theoretical and market prices.
    
    Args:
        strikes: List of strike prices
        theoretical_prices: List of theoretical option prices
        market_prices: List of market option prices
        option_type: Type of option ('call' or 'put')
        title: Chart title
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Determine colors based on option type
    color = 'b' if option_type.lower() == 'call' else 'r'
    
    # Plot prices on top subplot
    ax1.plot(strikes, theoretical_prices, f'{color}-', marker='o', label='Theoretical Prices')
    ax1.plot(strikes, market_prices, 'g-', marker='s', label='Market Prices')
    
    # Set labels and title for top subplot
    if title is None:
        title = f"{option_type.capitalize()} Option Price Comparison"
    ax1.set_title(title)
    ax1.set_xlabel('Strike Price ($)')
    ax1.set_ylabel('Option Price ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Calculate and plot price differences on bottom subplot
    price_diff = np.array(theoretical_prices) - np.array(market_prices)
    ax2.bar(strikes, price_diff, color=color, alpha=0.6)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Set labels for bottom subplot
    ax2.set_xlabel('Strike Price ($)')
    ax2.set_ylabel('Price Difference ($)')
    ax2.grid(True, alpha=0.3)
    
    # Add text boxes showing average difference and RMSE
    avg_diff = np.mean(price_diff)
    rmse = np.sqrt(np.mean(price_diff**2))
    
    text = f"Avg Diff: ${avg_diff:.4f}\nRMSE: ${rmse:.4f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.05, 0.95, text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_implied_volatility_smile(strikes: List[float],
                                 implied_vols: List[float],
                                 underlying_price: float,
                                 option_type: str = 'call',
                                 days_to_expiry: int = None,
                                 title: str = None,
                                 save_path: str = None) -> plt.Figure:
    """
    Plot the implied volatility smile.
    
    Args:
        strikes: List of strike prices
        implied_vols: List of implied volatilities
        underlying_price: Current stock price
        option_type: Type of option ('call' or 'put')
        days_to_expiry: Days to expiration (for the title)
        title: Chart title
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate moneyness (S/K ratio)
    moneyness = [underlying_price / strike for strike in strikes]
    
    # Determine color based on option type
    color = 'b' if option_type.lower() == 'call' else 'r'
    
    # Plot the implied volatility smile
    ax.plot(moneyness, implied_vols, f'{color}-', marker='o')
    
    # Mark the at-the-money point where moneyness = 1.0
    atm_index = None
    for i, m in enumerate(moneyness):
        if m >= 1.0:
            atm_index = i
            break
    
    if atm_index is not None:
        ax.plot(moneyness[atm_index], implied_vols[atm_index], 'go', markersize=10, 
               label='At-the-money')
    
    # Set labels and title
    ax.set_xlabel('Moneyness (S/K)')
    ax.set_ylabel('Implied Volatility')
    
    if title is None:
        title_parts = [f"{option_type.capitalize()} Option Implied Volatility Smile"]
        if days_to_expiry is not None:
            title_parts.append(f"({days_to_expiry} Days to Expiry)")
        title = " ".join(title_parts)
    
    ax.set_title(title)
    
    # Add grid and legend if ATM point is marked
    ax.grid(True, alpha=0.3)
    if atm_index is not None:
        ax.legend()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_volatility_surface(df: pd.DataFrame,
                           title: str = "Implied Volatility Surface",
                           save_path: str = None) -> plt.Figure:
    """
    Plot 3D implied volatility surface.
    
    Args:
        df: DataFrame with 'strike', 'daysToExpiry', and 'implied_volatility' columns
        title: Chart title
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Extract the required columns
    if not all(col in df.columns for col in ['strike', 'daysToExpiry', 'calculated_implied_volatility']):
        raise ValueError("DataFrame must contain 'strike', 'daysToExpiry', and 'calculated_implied_volatility' columns")
    
    # Create a pivot table for the surface
    pivot_df = df.pivot_table(
        values='calculated_implied_volatility', 
        index='strike', 
        columns='daysToExpiry'
    )
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the mesh grid
    x = pivot_df.columns  # days to expiry
    y = pivot_df.index    # strikes
    X, Y = np.meshgrid(x, y)
    Z = pivot_df.values
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.8)
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Implied Volatility')
    
    # Set labels and title
    ax.set_xlabel('Days to Expiry')
    ax.set_ylabel('Strike Price ($)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(title)
    
    # Adjust viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_historical_vs_implied_volatility(hist_dates: List[datetime],
                                         hist_volatility: List[float],
                                         implied_dates: List[datetime] = None,
                                         implied_volatility: List[float] = None,
                                         title: str = "Historical vs. Implied Volatility",
                                         save_path: str = None) -> plt.Figure:
    """
    Plot comparison of historical and implied volatility over time.
    
    Args:
        hist_dates: List of dates for historical volatility
        hist_volatility: List of historical volatility values
        implied_dates: List of dates for implied volatility
        implied_volatility: List of implied volatility values
        title: Chart title
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot historical volatility
    ax.plot(hist_dates, hist_volatility, 'b-', label='Historical Volatility')
    
    # Plot implied volatility if provided
    if implied_dates is not None and implied_volatility is not None:
        ax.plot(implied_dates, implied_volatility, 'r-', label='Implied Volatility')
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')
    ax.set_title(title)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()  # Rotate date labels
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_price_differences_heatmap(comparison_df: pd.DataFrame,
                                  value_col: str = 'price_diff_pct',
                                  title: str = "Option Price Differences Heatmap",
                                  save_path: str = None) -> plt.Figure:
    """
    Create a heatmap of price differences by strike and expiration.
    
    Args:
        comparison_df: DataFrame with comparison results
        value_col: Column to use for the heatmap values
        title: Chart title
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Create a pivot table for the heatmap
    pivot_df = comparison_df.pivot_table(
        values=value_col,
        index='strike',
        columns='daysToExpiry',
        aggfunc='mean'
    )
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the heatmap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(pivot_df, ax=ax, cmap=cmap, center=0, annot=True, fmt='.2f',
               linewidths=.5, cbar_kws={'label': f'{value_col} (%)' if 'pct' in value_col else value_col})
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel('Days to Expiry')
    ax.set_ylabel('Strike Price ($)')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_moneyness_analysis(moneyness_analysis: Dict[str, Dict[str, float]],
                           metric: str = 'avg_diff_pct',
                           title: str = None,
                           save_path: str = None) -> plt.Figure:
    """
    Plot analysis results by option moneyness.
    
    Args:
        moneyness_analysis: Dictionary with analysis by moneyness category
        metric: Metric to plot ('avg_diff', 'avg_diff_pct', 'MAE', 'RMSE')
        title: Chart title
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Extract data for plotting
    categories = list(moneyness_analysis.keys())
    values = [moneyness_analysis[cat][metric] for cat in categories]
    counts = [moneyness_analysis[cat]['count'] for cat in categories]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar colors based on metric values
    colors = ['g' if v > 0 else 'r' for v in values]
    
    # Plot the bars
    bars = ax.bar(categories, values, alpha=0.7, color=colors)
    
    # Add count labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        y_pos = height + 0.1 if height > 0 else height - 0.1
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'n={count}', ha='center', va='bottom' if height > 0 else 'top')
    
    # Set labels and title
    ax.set_xlabel('Option Moneyness')
    
    # Set y-label based on metric
    if metric == 'avg_diff':
        ax.set_ylabel('Average Price Difference ($)')
    elif metric == 'avg_diff_pct':
        ax.set_ylabel('Average Price Difference (%)')
    elif metric in ['MAE', 'RMSE']:
        ax.set_ylabel(f'{metric} ($)')
    else:
        ax.set_ylabel(metric)
    
    if title is None:
        metric_name = {
            'avg_diff': 'Average Price Difference',
            'avg_diff_pct': 'Average Price Difference (%)',
            'MAE': 'Mean Absolute Error',
            'RMSE': 'Root Mean Square Error'
        }.get(metric, metric)
        title = f"{metric_name} by Option Moneyness"
    
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add zero line if appropriate
    if min(values) < 0 and max(values) > 0:
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_multi_expiry_price_curves(options_data: pd.DataFrame,
                                 expirations: List[int] = None,
                                 option_type: str = 'call',
                                 title: str = None,
                                 save_path: str = None) -> plt.Figure:
    """
    Plot option price curves for multiple expiration dates.
    
    Args:
        options_data: DataFrame with options data
        expirations: List of days to expiry to include
        option_type: Type of option ('call' or 'put')
        title: Chart title
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Filter data for the specified option type
    filtered_data = options_data[options_data['option_type'] == option_type].copy()
    
    # Use all expirations if none specified
    if expirations is None:
        expirations = sorted(filtered_data['daysToExpiry'].unique())
    else:
        # Filter for the specified expirations
        filtered_data = filtered_data[filtered_data['daysToExpiry'].isin(expirations)]
    
    # Get the underlying price
    underlying_price = filtered_data['underlying_price'].iloc[0]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each expiration curve
    for days in expirations:
        # Get data for this expiration
        expiry_data = filtered_data[filtered_data['daysToExpiry'] == days]
        
        # Sort by strike
        expiry_data = expiry_data.sort_values('strike')
        
        # Plot the curve
        ax.plot(expiry_data['strike'], expiry_data['theoretical_price'], 
               marker='o', label=f'{days} Days to Expiry')
    
    # Mark the underlying price
    ax.axvline(x=underlying_price, color='k', linestyle='--', alpha=0.5, 
              label=f'Underlying Price (${underlying_price:.2f})')
    
    # Set labels and title
    ax.set_xlabel('Strike Price ($)')
    ax.set_ylabel('Option Price ($)')
    
    if title is None:
        title = f"{option_type.capitalize()} Option Prices by Expiration"
    ax.set_title(title)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    """Test the visualization functions with sample data."""
    
    # Create sample data for testing
    import numpy as np
    from src.models.black_scholes import bs_price
    
    # Parameters
    S = 100.0  # Current stock price
    strikes = np.arange(80, 121, 5)  # Strike prices
    T = 1.0  # Time to expiry (1 year)
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    
    # Calculate theoretical option prices
    call_prices = [bs_price(S, K, T, r, sigma, 'call') for K in strikes]
    put_prices = [bs_price(S, K, T, r, sigma, 'put') for K in strikes]
    
    # Create "market" prices with some random noise
    call_market = [price * (1 + np.random.uniform(-0.1, 0.1)) for price in call_prices]
    put_market = [price * (1 + np.random.uniform(-0.1, 0.1)) for price in put_prices]
    
    # Test option price curve plot
    fig1 = plot_option_price_curve(strikes, call_prices, put_prices, S)
    plt.show()
    
    # Test price comparison plot
    fig2 = plot_price_comparison(strikes, call_prices, call_market, 'call')
    plt.show()
    
    # Create sample implied volatility data
    implied_vols = [0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.175, 0.18, 0.19]
    
    # Test implied volatility smile plot
    fig3 = plot_implied_volatility_smile(strikes, implied_vols, S, 'call', 30)
    plt.show()
    
    # Create sample historical volatility data
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(30, 0, -1)]
    hist_vol = [0.2 + 0.02 * np.sin(i/5) for i in range(30)]
    
    # Test historical volatility plot
    fig4 = plot_historical_vs_implied_volatility(dates, hist_vol)
    plt.show()
    
    print("Visualization tests completed successfully!")

# Add this to src/visualization/option_charts.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional

def plot_option_price_curve(strikes: List[float], 
                          call_prices: List[float], 
                          put_prices: List[float] = None,
                          underlying_price: float = None,
                          title: str = "Option Price Curve",
                          save_path: str = None) -> plt.Figure:
    """
    Plot option prices against strike prices.
    
    Args:
        strikes: List of strike prices
        call_prices: List of call option prices
        put_prices: List of put option prices (optional)
        underlying_price: Current stock price (to mark on the x-axis)
        title: Chart title
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot call prices
    ax.plot(strikes, call_prices, 'b-', marker='o', label='Call Prices')
    
    # Plot put prices if provided
    if put_prices is not None:
        ax.plot(strikes, put_prices, 'r-', marker='s', label='Put Prices')
    
    # Mark the underlying price if provided
    if underlying_price is not None:
        ax.axvline(x=underlying_price, color='k', linestyle='--', alpha=0.5, 
                  label=f'Underlying Price (${underlying_price:.2f})')
    
    # Set labels and title
    ax.set_xlabel('Strike Price ($)')
    ax.set_ylabel('Option Price ($)')
    ax.set_title(title)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig    