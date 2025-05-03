# src/visualization/greek_charts.py
"""
Functions for visualizing option Greeks.

This module provides tools to create various charts for visualizing option Greeks,
including delta, gamma, theta, vega, and rho across different parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict, List, Tuple, Any, Optional
import seaborn as sns

import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Import Greek calculation functions
from src.analytics.greeks import delta, gamma, theta, vega, rho, calculate_all_greeks


def plot_delta_curve(strikes: List[float],
                    call_deltas: List[float],
                    put_deltas: List[float] = None,
                    underlying_price: float = None,
                    title: str = "Option Delta Curve",
                    save_path: str = None) -> plt.Figure:
    """
    Plot option delta values against strike prices.
    
    Args:
        strikes: List of strike prices
        call_deltas: List of call option delta values
        put_deltas: List of put option delta values (optional)
        underlying_price: Current stock price (to mark on the x-axis)
        title: Chart title
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot call deltas
    ax.plot(strikes, call_deltas, 'b-', marker='o', label='Call Delta')
    
    # Plot put deltas if provided
    if put_deltas is not None:
        ax.plot(strikes, put_deltas, 'r-', marker='s', label='Put Delta')
    
    # Mark the underlying price if provided
    if underlying_price is not None:
        ax.axvline(x=underlying_price, color='k', linestyle='--', alpha=0.5, 
                  label=f'Underlying Price (${underlying_price:.2f})')
    
    # Add horizontal lines at 0, 0.5, and 1.0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax.axhline(y=0.5, color='k', linestyle=':', alpha=0.2)
    ax.axhline(y=-0.5, color='k', linestyle=':', alpha=0.2)
    ax.axhline(y=1, color='k', linestyle=':', alpha=0.2)
    ax.axhline(y=-1, color='k', linestyle=':', alpha=0.2)
    
    # Set labels and title
    ax.set_xlabel('Strike Price ($)')
    ax.set_ylabel('Delta')
    ax.set_title(title)
    
    # Set y-axis limits
    ax.set_ylim(-1.1, 1.1)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_gamma_curve(strikes: List[float],
                    gammas: List[float],
                    underlying_price: float = None,
                    title: str = "Option Gamma Curve",
                    save_path: str = None) -> plt.Figure:
    """
    Plot option gamma values against strike prices.
    
    Args:
        strikes: List of strike prices
        gammas: List of option gamma values
        underlying_price: Current stock price (to mark on the x-axis)
        title: Chart title
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot gammas
    ax.plot(strikes, gammas, 'g-', marker='o', label='Gamma')
    
    # Mark the underlying price if provided
    if underlying_price is not None:
        ax.axvline(x=underlying_price, color='k', linestyle='--', alpha=0.5, 
                  label=f'Underlying Price (${underlying_price:.2f})')
    
    # Set labels and title
    ax.set_xlabel('Strike Price ($)')
    ax.set_ylabel('Gamma')
    ax.set_title(title)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_theta_curve(strikes: List[float],
                    call_thetas: List[float],
                    put_thetas: List[float] = None,
                    underlying_price: float = None,
                    title: str = "Option Theta Curve",
                    daily: bool = True,
                    save_path: str = None) -> plt.Figure:
    """
    Plot option theta values against strike prices.
    
    Args:
        strikes: List of strike prices
        call_thetas: List of call option theta values
        put_thetas: List of put option theta values (optional)
        underlying_price: Current stock price (to mark on the x-axis)
        title: Chart title
        daily: If True, display daily theta values (theta/365)
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to daily values if requested
    scale_factor = 1.0
    y_label = "Theta (annual)"
    if daily:
        scale_factor = 1/365
        y_label = "Theta (daily)"
    
    # Plot call thetas
    call_values = [theta * scale_factor for theta in call_thetas]
    ax.plot(strikes, call_values, 'b-', marker='o', label='Call Theta')
    
    # Plot put thetas if provided
    if put_thetas is not None:
        put_values = [theta * scale_factor for theta in put_thetas]
        ax.plot(strikes, put_values, 'r-', marker='s', label='Put Theta')
    
    # Mark the underlying price if provided
    if underlying_price is not None:
        ax.axvline(x=underlying_price, color='k', linestyle='--', alpha=0.5, 
                  label=f'Underlying Price (${underlying_price:.2f})')
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Set labels and title
    ax.set_xlabel('Strike Price ($)')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_vega_curve(strikes: List[float],
                   vegas: List[float],
                   underlying_price: float = None,
                   title: str = "Option Vega Curve",
                   save_path: str = None) -> plt.Figure:
    """
    Plot option vega values against strike prices.
    
    Args:
        strikes: List of strike prices
        vegas: List of option vega values
        underlying_price: Current stock price (to mark on the x-axis)
        title: Chart title
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot vegas
    ax.plot(strikes, vegas, 'purple', marker='o', label='Vega')
    
    # Mark the underlying price if provided
    if underlying_price is not None:
        ax.axvline(x=underlying_price, color='k', linestyle='--', alpha=0.5, 
                  label=f'Underlying Price (${underlying_price:.2f})')
    
    # Set labels and title
    ax.set_xlabel('Strike Price ($)')
    ax.set_ylabel('Vega (per 1% change in volatility)')
    ax.set_title(title)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_rho_curve(strikes: List[float],
                  call_rhos: List[float],
                  put_rhos: List[float] = None,
                  underlying_price: float = None,
                  title: str = "Option Rho Curve",
                  save_path: str = None) -> plt.Figure:
    """
    Plot option rho values against strike prices.
    
    Args:
        strikes: List of strike prices
        call_rhos: List of call option rho values
        put_rhos: List of put option rho values (optional)
        underlying_price: Current stock price (to mark on the x-axis)
        title: Chart title
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot call rhos
    ax.plot(strikes, call_rhos, 'b-', marker='o', label='Call Rho')
    
    # Plot put rhos if provided
    if put_rhos is not None:
        ax.plot(strikes, put_rhos, 'r-', marker='s', label='Put Rho')
    
    # Mark the underlying price if provided
    if underlying_price is not None:
        ax.axvline(x=underlying_price, color='k', linestyle='--', alpha=0.5, 
                  label=f'Underlying Price (${underlying_price:.2f})')
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Set labels and title
    ax.set_xlabel('Strike Price ($)')
    ax.set_ylabel('Rho (per 1% change in interest rate)')
    ax.set_title(title)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_all_greeks(strikes: List[float],
                   greeks_data: Dict[str, List[float]],
                   underlying_price: float = None,
                   option_type: str = 'call',
                   save_dir: str = None) -> Dict[str, plt.Figure]:
    """
    Plot all Greeks against strike prices.
    
    Args:
        strikes: List of strike prices
        greeks_data: Dictionary with lists of Greek values
        underlying_price: Current stock price
        option_type: Type of option ('call' or 'put')
        save_dir: Directory to save the figures (optional)
        
    Returns:
        Dictionary with all Figure objects
    """
    figures = {}
    
    # Ensure save directory exists if provided
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Plot Delta
    if 'delta' in greeks_data:
        title = f"{option_type.capitalize()} Option Delta Curve"
        save_path = os.path.join(save_dir, f"{option_type}_delta.png") if save_dir else None
        figures['delta'] = plot_delta_curve(strikes, greeks_data['delta'], None, underlying_price, title, save_path)
    
    # Plot Gamma
    if 'gamma' in greeks_data:
        title = f"Option Gamma Curve"
        save_path = os.path.join(save_dir, "gamma.png") if save_dir else None
        figures['gamma'] = plot_gamma_curve(strikes, greeks_data['gamma'], underlying_price, title, save_path)
    
    # Plot Theta
    if 'theta' in greeks_data:
        title = f"{option_type.capitalize()} Option Theta Curve"
        save_path = os.path.join(save_dir, f"{option_type}_theta.png") if save_dir else None
        figures['theta'] = plot_theta_curve(strikes, greeks_data['theta'], None, underlying_price, title, True, save_path)
    
    # Plot Vega
    if 'vega' in greeks_data:
        title = f"Option Vega Curve"
        save_path = os.path.join(save_dir, "vega.png") if save_dir else None
        figures['vega'] = plot_vega_curve(strikes, greeks_data['vega'], underlying_price, title, save_path)
    
    # Plot Rho
    if 'rho' in greeks_data:
        title = f"{option_type.capitalize()} Option Rho Curve"
        save_path = os.path.join(save_dir, f"{option_type}_rho.png") if save_dir else None
        figures['rho'] = plot_rho_curve(strikes, greeks_data['rho'], None, underlying_price, title, save_path)
    
    return figures


def plot_greeks_by_time(days_to_expiry: List[int],
                       greeks_by_expiry: Dict[str, List[List[float]]],
                       strike: float = None,
                       underlying_price: float = None,
                       option_type: str = 'call',
                       save_dir: str = None) -> Dict[str, plt.Figure]:
    """
    Plot Greeks against time to expiration for different strikes.
    
    Args:
        days_to_expiry: List of days to expiration
        greeks_by_expiry: Dictionary with lists of Greek values for each expiry
        strike: Strike price to mark on the plot
        underlying_price: Current stock price
        option_type: Type of option ('call' or 'put')
        save_dir: Directory to save the figures (optional)
        
    Returns:
        Dictionary with all Figure objects
    """
    figures = {}
    
    # Ensure save directory exists if provided
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Convert days to years for x-axis
    years_to_expiry = [days / 365 for days in days_to_expiry]
    
    # Plot each Greek
    for greek, values_by_expiry in greeks_by_expiry.items():
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get colors for multiple lines
        colors = plt.cm.viridis(np.linspace(0, 1, len(values_by_expiry)))
        
        # Plot each strike line
        for i, (strike_value, values) in enumerate(values_by_expiry.items()):
            ax.plot(years_to_expiry, values, color=colors[i], marker='o', 
                   label=f'Strike=${strike_value}')
        
        # Mark the selected strike if provided
        if strike is not None:
            ax.axvline(x=strike, color='k', linestyle='--', alpha=0.5, 
                      label=f'Strike=${strike:.2f}')
        
        # Set labels and title
        ax.set_xlabel('Time to Expiry (years)')
        
        # Set y-label based on Greek
        if greek == 'delta':
            ax.set_ylabel('Delta')
        elif greek == 'gamma':
            ax.set_ylabel('Gamma')
        elif greek == 'theta':
            ax.set_ylabel('Theta (annual)')
        elif greek == 'vega':
            ax.set_ylabel('Vega (per 1% change in volatility)')
        elif greek == 'rho':
            ax.set_ylabel('Rho (per 1% change in interest rate)')
        else:
            ax.set_ylabel(greek.capitalize())
        
        ax.set_title(f"{option_type.capitalize()} Option {greek.capitalize()} by Time to Expiry")
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Save if directory provided
        if save_dir:
            save_path = os.path.join(save_dir, f"{option_type}_{greek}_vs_time.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        figures[greek] = fig
    
    return figures


def plot_greeks_by_volatility(volatilities: List[float],
                             greeks_by_vol: Dict[str, List[float]],
                             title: str = None,
                             option_type: str = 'call',
                             save_path: str = None) -> plt.Figure:
    """
    Plot Greeks against volatility.
    
    Args:
        volatilities: List of volatility values
        greeks_by_vol: Dictionary with lists of Greek values for each volatility
        title: Chart title
        option_type: Type of option ('call' or 'put')
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each Greek
    for greek, values in greeks_by_vol.items():
        # Normalize values for plotting on the same scale
        if greek == 'delta':
            normalized = values
            label = 'Delta'
        elif greek == 'gamma':
            # Scale gamma up
            normalized = [val * 10 for val in values]
            label = 'Gamma × 10'
        elif greek == 'theta':
            # Convert to daily and flip sign for clearer visualization
            normalized = [-val / 365 for val in values]
            label = 'Theta (daily)'
        elif greek == 'vega':
            normalized = values
            label = 'Vega'
        elif greek == 'rho':
            normalized = values
            label = 'Rho'
        else:
            normalized = values
            label = greek.capitalize()
        
        ax.plot(volatilities, normalized, marker='o', label=label)
    
    # Set labels and title
    vol_pct = [vol * 100 for vol in volatilities]
    ax.set_xlabel('Volatility (%)')
    ax.set_ylabel('Greek Value (scaled)')
    
    if title is None:
        title = f"{option_type.capitalize()} Option Greeks by Volatility"
    ax.set_title(title)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
def plot_3d_greek_surface(strikes: List[float],
                         days_to_expiry: List[int],
                         greek_values: List[List[float]],
                         greek_name: str = 'delta',
                         option_type: str = 'call',
                         title: str = None,
                         save_path: str = None) -> plt.Figure:
    """
    Create a 3D surface plot for a Greek across strikes and expiration dates.
    
    Args:
        strikes: List of strike prices
        days_to_expiry: List of days to expiration
        greek_values: 2D array of Greek values [strike_idx][expiry_idx]
        greek_name: Name of the Greek being plotted
        option_type: Type of option ('call' or 'put')
        title: Chart title
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Convert days to years for better visualization
    years_to_expiry = [days / 365 for days in days_to_expiry]
    
    # Create meshgrid for 3D plot
    X, Y = np.meshgrid(years_to_expiry, strikes)
    Z = np.array(greek_values)
    
    # Create the figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the surface plot
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.8)
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=f'{greek_name.capitalize()} Value')
    
    # Set labels and title
    ax.set_xlabel('Time to Expiry (years)')
    ax.set_ylabel('Strike Price ($)')
    ax.set_zlabel(f'{greek_name.capitalize()} Value')
    
    if title is None:
        title = f"{option_type.capitalize()} Option {greek_name.capitalize()} Surface"
    ax.set_title(title)
    
    # Adjust viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_greek_vs_price_change(underlying_prices: List[float],
                             initial_price: float,
                             greeks_by_price: Dict[str, List[float]],
                             option_type: str = 'call',
                             title: str = None,
                             save_path: str = None) -> plt.Figure:
    """
    Plot Greeks against changes in underlying price.
    
    Args:
        underlying_prices: List of possible underlying prices
        initial_price: Initial price of the underlying
        greeks_by_price: Dictionary with lists of Greek values for each price
        option_type: Type of option ('call' or 'put')
        title: Chart title
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate percentage changes from initial price
    pct_changes = [(price - initial_price) / initial_price * 100 for price in underlying_prices]
    
    # Plot each Greek
    for greek, values in greeks_by_price.items():
        # Normalize values for plotting on the same scale
        if greek == 'delta':
            normalized = values
            label = 'Delta'
        elif greek == 'gamma':
            # Scale gamma up
            normalized = [val * 10 for val in values]
            label = 'Gamma × 10'
        elif greek == 'theta':
            # Convert to daily
            normalized = [val / 365 for val in values]
            label = 'Theta (daily)'
        elif greek == 'vega':
            normalized = values
            label = 'Vega'
        elif greek == 'rho':
            normalized = values
            label = 'Rho'
        else:
            normalized = values
            label = greek.capitalize()
        
        ax.plot(pct_changes, normalized, marker='o', label=label)
    
    # Mark the zero change point
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, 
              label='Initial Price')
    
    # Set labels and title
    ax.set_xlabel('Price Change (%)')
    ax.set_ylabel('Greek Value (scaled)')
    
    if title is None:
        title = f"{option_type.capitalize()} Option Greeks by Underlying Price Change"
    ax.set_title(title)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_greek_heatmap(strikes: List[float],
                      days_to_expiry: List[int],
                      greek_values: List[List[float]],
                      greek_name: str = 'delta',
                      option_type: str = 'call',
                      title: str = None,
                      save_path: str = None) -> plt.Figure:
    """
    Create a heatmap for a Greek across strikes and expiration dates.
    
    Args:
        strikes: List of strike prices
        days_to_expiry: List of days to expiration
        greek_values: 2D array of Greek values [strike_idx][expiry_idx]
        greek_name: Name of the Greek being plotted
        option_type: Type of option ('call' or 'put')
        title: Chart title
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Convert to a DataFrame for easier plotting
    df = pd.DataFrame(greek_values, index=strikes, columns=days_to_expiry)
    
    # Choose an appropriate colormap based on the Greek
    if greek_name == 'delta':
        # Delta ranges from -1 to 1, use a diverging colormap
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        center = 0
    elif greek_name == 'theta':
        # Theta is often negative, use a sequential colormap
        cmap = 'Blues_r'  # _r for reversed (more negative = darker)
        center = None
    else:
        # Other Greeks typically use a sequential colormap
        cmap = 'viridis'
        center = None
    
    # Create the heatmap
    sns.heatmap(df, ax=ax, cmap=cmap, center=center, annot=True, fmt='.4f',
               linewidths=.5, cbar_kws={'label': f'{greek_name.capitalize()} Value'})
    
    # Set labels and title
    ax.set_xlabel('Days to Expiry')
    ax.set_ylabel('Strike Price ($)')
    
    if title is None:
        title = f"{option_type.capitalize()} Option {greek_name.capitalize()} Heatmap"
    ax.set_title(title)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_all_greek_charts(options_data: pd.DataFrame,
                             option_type: str = 'call',
                             save_dir: str = None) -> Dict[str, plt.Figure]:
    """
    Generate a complete set of Greek charts for the given options data.
    
    Args:
        options_data: DataFrame with options data
        option_type: Type of option ('call' or 'put')
        save_dir: Directory to save the figures (optional)
        
    Returns:
        Dictionary with all Figure objects
    """
    # Filter for the specified option type
    filtered_data = options_data[options_data['option_type'] == option_type].copy()
    
    # Get the underlying price and parameters
    underlying_price = filtered_data['underlying_price'].iloc[0]
    risk_free_rate = filtered_data['risk_free_rate'].iloc[0]
    # Replace with this more robust approach
    if 'historical_volatility' in filtered_data.columns:
        volatility = filtered_data['historical_volatility'].iloc[0]
    elif 'volatility' in filtered_data.columns:
        volatility = filtered_data['volatility'].iloc[0]
    elif 'impliedVolatility' in filtered_data.columns:
        volatility = filtered_data['impliedVolatility'].iloc[0]
    elif 'calculated_implied_volatility' in filtered_data.columns:
        volatility = filtered_data['calculated_implied_volatility'].iloc[0]
    else:
    # Use default volatility if no volatility column is found
        volatility = 0.2
        print(f"Warning: No volatility data found in options data, using default value of 20%")
    
    # Get unique strikes and days to expiry
    strikes = sorted(filtered_data['strike'].unique())
    days_to_expiry = sorted(filtered_data['daysToExpiry'].unique())
    
    # Create a dictionary to store all figures
    figures = {}
    
    # Calculate Greek values for plotting vs. strike
    greek_data = {}
    for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
        greek_data[greek] = []
    
    # Use a single expiry for the strike curves
    if len(days_to_expiry) > 0:
        mid_expiry = days_to_expiry[len(days_to_expiry)//2]
        T = mid_expiry / 365
        
        for K in strikes:
            # Calculate all Greeks
            params = {
                'S': underlying_price,
                'K': K,
                'T': T,
                'r': risk_free_rate,
                'sigma': volatility,
                'option_type': option_type
            }
            all_greeks = calculate_all_greeks(params)
            
            # Store values
            for greek in greek_data:
                greek_data[greek].append(all_greeks[greek])
        
        # Create individual Greek vs. strike plots
        title = f"{option_type.capitalize()} Option Greeks ({mid_expiry} Days to Expiry)"
        save_path = os.path.join(save_dir, f"{option_type}_greeks_strike.png") if save_dir else None
        figures['greeks_vs_strike'] = plot_all_greeks(strikes, greek_data, underlying_price, option_type, save_dir)
    
    # Calculate Greeks for varying expiry
    if len(strikes) > 0:
        # Use an at-the-money strike
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))
        atm_strike = strikes[atm_idx]
        
        # Calculate Greeks for different expiries
        greek_by_time = {greek: [] for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']}
        
        for days in days_to_expiry:
            T = days / 365
            
            # Calculate all Greeks
            params = {
                'S': underlying_price,
                'K': atm_strike,
                'T': T,
                'r': risk_free_rate,
                'sigma': volatility,
                'option_type': option_type
            }
            all_greeks = calculate_all_greeks(params)
            
            # Store values
            for greek in greek_by_time:
                greek_by_time[greek].append(all_greeks[greek])
        
        # Create Greek vs. time charts
        title = f"{option_type.capitalize()} Option Greeks Over Time (Strike=${atm_strike})"
        save_path = os.path.join(save_dir, f"{option_type}_greeks_time.png") if save_dir else None
        
        # Plot each Greek vs. time
        for greek, values in greek_by_time.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(days_to_expiry, values, marker='o')
            ax.set_xlabel('Days to Expiry')
            ax.set_ylabel(f'{greek.capitalize()} Value')
            ax.set_title(f"{option_type.capitalize()} Option {greek.capitalize()} vs Time")
            ax.grid(True, alpha=0.3)
            
            if save_dir:
                save_path = os.path.join(save_dir, f"{option_type}_{greek}_time.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            figures[f'{greek}_vs_time'] = fig
    
    # Create 3D surface for delta (if multiple expiries and strikes)
    if len(strikes) > 1 and len(days_to_expiry) > 1:
        # Calculate delta values for all combinations
        delta_values = []
        
        for K in strikes:
            row = []
            for days in days_to_expiry:
                T = days / 365
                
                # Calculate delta
                delta_val = delta(
                    underlying_price, K, T, risk_free_rate, volatility, option_type
                )
                row.append(delta_val)
            
            delta_values.append(row)
        
        # Create 3D surface plot
        save_path = os.path.join(save_dir, f"{option_type}_delta_surface.png") if save_dir else None
        figures['delta_surface'] = plot_3d_greek_surface(
            strikes, days_to_expiry, delta_values, 'delta', option_type, None, save_path
        )
        
        # Create heatmap
        save_path = os.path.join(save_dir, f"{option_type}_delta_heatmap.png") if save_dir else None
        figures['delta_heatmap'] = plot_greek_heatmap(
            strikes, days_to_expiry, delta_values, 'delta', option_type, None, save_path
        )
    
    # Calculate Greeks for varying underlying price
    price_range = np.linspace(underlying_price * 0.8, underlying_price * 1.2, 11)
    
    if len(days_to_expiry) > 0:
        # Use a medium-term expiry
        mid_expiry = days_to_expiry[len(days_to_expiry)//2]
        T = mid_expiry / 365
        
        # Use an at-the-money strike
        if len(strikes) > 0:
            atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))
            atm_strike = strikes[atm_idx]
            
            # Calculate Greeks for different prices
            greeks_by_price = {greek: [] for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']}
            
            for S in price_range:
                # Calculate all Greeks
                params = {
                    'S': S,
                    'K': atm_strike,
                    'T': T,
                    'r': risk_free_rate,
                    'sigma': volatility,
                    'option_type': option_type
                }
                all_greeks = calculate_all_greeks(params)
                
                # Store values
                for greek in greeks_by_price:
                    greeks_by_price[greek].append(all_greeks[greek])
            
            # Create Greek vs. price chart
            save_path = os.path.join(save_dir, f"{option_type}_greeks_price.png") if save_dir else None
            figures['greeks_vs_price'] = plot_greek_vs_price_change(
                price_range, underlying_price, greeks_by_price, option_type, None, save_path
            )
    
    return figures


if __name__ == "__main__":
    """Test the Greek visualization functions with sample data."""
    
    # Create sample data for testing
    import numpy as np
    from src.models.black_scholes import bs_price
    
    # Parameters
    S = 100.0  # Current stock price
    strikes = np.arange(80, 121, 5)  # Strike prices
    T = 1.0  # Time to expiry (1 year)
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    
    # Calculate Greek values
    call_deltas = [delta(S, K, T, r, sigma, 'call') for K in strikes]
    put_deltas = [delta(S, K, T, r, sigma, 'put') for K in strikes]
    gammas = [gamma(S, K, T, r, sigma) for K in strikes]
    call_thetas = [theta(S, K, T, r, sigma, 'call') for K in strikes]
    put_thetas = [theta(S, K, T, r, sigma, 'put') for K in strikes]
    vegas = [vega(S, K, T, r, sigma) for K in strikes]
    call_rhos = [rho(S, K, T, r, sigma, 'call') for K in strikes]
    put_rhos = [rho(S, K, T, r, sigma, 'put') for K in strikes]
    
    # Test delta curve plot
    fig1 = plot_delta_curve(strikes, call_deltas, put_deltas, S)
    plt.show()
    
    # Test gamma curve plot
    fig2 = plot_gamma_curve(strikes, gammas, S)
    plt.show()
    
    # Test theta curve plot
    fig3 = plot_theta_curve(strikes, call_thetas, put_thetas, S, daily=True)
    plt.show()
    
    # Test vega curve plot
    fig4 = plot_vega_curve(strikes, vegas, S)
    plt.show()
    
    # Test rho curve plot
    fig5 = plot_rho_curve(strikes, call_rhos, put_rhos, S)
    plt.show()
    
    # Test plotting all Greeks
    greeks_data = {
        'delta': call_deltas,
        'gamma': gammas,
        'theta': call_thetas,
        'vega': vegas,
        'rho': call_rhos
    }
    
    all_figs = plot_all_greeks(strikes, greeks_data, S, 'call')
    for greek, fig in all_figs.items():
        plt.figure(fig.number)
        plt.show()
    
    # Test 3D Greek surface
    days_list = [30, 60, 90, 180, 270, 365]
    T_list = [d/365 for d in days_list]
    
    # Calculate delta values for the surface
    delta_values = []
    for K in strikes:
        row = []
        for T_val in T_list:
            delta_val = delta(S, K, T_val, r, sigma, 'call')
            row.append(delta_val)
        delta_values.append(row)
    
    # Test 3D surface plot
    fig6 = plot_3d_greek_surface(strikes, days_list, delta_values, 'delta', 'call')
    plt.show()
    
    # Test heatmap
    fig7 = plot_greek_heatmap(strikes, days_list, delta_values, 'delta', 'call')
    plt.show()
    
    print("Greek visualization tests completed successfully!")