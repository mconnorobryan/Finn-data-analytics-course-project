# src/analytics/comparison.py
"""
Functions for comparing theoretical option prices with market prices.

This module provides tools to analyze the difference between Black-Scholes
theoretical prices and actual market prices, helping identify potential
mispricing or arbitrage opportunities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from datetime import datetime

import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Import the necessary modules
from src.models.black_scholes import bs_price, implied_volatility
from src.analytics.greeks import calculate_all_greeks


def calculate_price_differences(options_data: pd.DataFrame,
                               pricing_model: callable = bs_price,
                               volatility_source: str = 'historical',
                               risk_free_rate: float = None) -> pd.DataFrame:
    """
    Calculate the difference between theoretical and market prices.
    
    Args:
        options_data: DataFrame with options data, must include columns:
                     'option_type', 'strike', 'T', 'lastPrice', etc.
        pricing_model: Function to calculate theoretical price
        volatility_source: Source of volatility ('historical' or 'implied')
        risk_free_rate: Risk-free interest rate (if None, use from data)
        
    Returns:
        DataFrame with added columns for theoretical prices and differences
    """
    # Make a copy to avoid modifying the original
    result_df = options_data.copy()
    
    # Use provided risk-free rate or extract from data
    if risk_free_rate is None:
        if 'risk_free_rate' in result_df.columns:
            risk_free_rate = result_df['risk_free_rate'].iloc[0]
        else:
            risk_free_rate = 0.05  # Default if not provided
    
    # Calculate theoretical prices and differences
    theoretical_prices = []
    price_diffs = []
    price_diff_pcts = []
    
    for _, row in result_df.iterrows():
        # Extract parameters
        S = row.get('underlying_price', row.get('S', 0))
        K = row.get('strike', row.get('K', 0))
        T = row.get('T', row.get('time_to_expiry', 0))
        
        # Get volatility based on specified source
        if volatility_source == 'implied' and 'impliedVolatility' in row:
            sigma = row['impliedVolatility']
        else:
            sigma = row.get('historical_volatility', row.get('volatility', 0.2))
        
        option_type = row.get('option_type', 'call').lower()
        
        # Calculate theoretical price
        theo_price = pricing_model(S, K, T, risk_free_rate, sigma, option_type)
        
        # Get market price
        market_price = row.get('lastPrice', row.get('market_price', 0))
        
        # Calculate differences
        price_diff = theo_price - market_price
        price_diff_pct = 100 * price_diff / market_price if market_price > 0 else float('nan')
        
        # Store values
        theoretical_prices.append(theo_price)
        price_diffs.append(price_diff)
        price_diff_pcts.append(price_diff_pct)
    
    # Add calculated columns to the DataFrame
    result_df['theoretical_price'] = theoretical_prices
    result_df['price_diff'] = price_diffs
    result_df['price_diff_pct'] = price_diff_pcts
    
    return result_df


def calculate_implied_volatility_surface(options_data: pd.DataFrame,
                                        pricing_model: callable = bs_price) -> pd.DataFrame:
    """
    Calculate implied volatility for all options and create a volatility surface.
    
    Args:
        options_data: DataFrame with options data
        pricing_model: Function to calculate theoretical price
        
    Returns:
        DataFrame with added implied volatility column
    """
    # Make a copy to avoid modifying the original
    result_df = options_data.copy()
    
    # Calculate implied volatility for each option
    impl_vols = []
    
    for _, row in result_df.iterrows():
        # Extract parameters
        S = row.get('underlying_price', row.get('S', 0))
        K = row.get('strike', row.get('K', 0))
        T = row.get('T', row.get('time_to_expiry', 0))
        r = row.get('risk_free_rate', 0.05)
        market_price = row.get('lastPrice', row.get('market_price', 0))
        option_type = row.get('option_type', 'call').lower()
        
        # Calculate implied volatility
        if market_price > 0 and T > 0:
            impl_vol = implied_volatility(market_price, S, K, T, r, option_type)
            if impl_vol is None:
                impl_vol = float('nan')
        else:
            impl_vol = float('nan')
            
        impl_vols.append(impl_vol)
    
    # Add implied volatility column
    result_df['calculated_implied_volatility'] = impl_vols
    
    return result_df


def identify_mispriced_options(comparison_df: pd.DataFrame,
                              threshold_pct: float = 5.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify potentially mispriced options based on price difference percentage.
    
    Args:
        comparison_df: DataFrame with comparison results
        threshold_pct: Threshold percentage difference to consider mispriced
        
    Returns:
        Tuple of (overpriced options, underpriced options)
    """
    # Filter for valid price differences (non-NaN)
    valid_df = comparison_df.dropna(subset=['price_diff_pct'])
    
    # Identify overpriced options (market > theoretical)
    overpriced = valid_df[valid_df['price_diff_pct'] < -threshold_pct].copy()
    
    # Identify underpriced options (theoretical > market)
    underpriced = valid_df[valid_df['price_diff_pct'] > threshold_pct].copy()
    
    # Sort by absolute percentage difference
    overpriced = overpriced.sort_values('price_diff_pct')
    underpriced = underpriced.sort_values('price_diff_pct', ascending=False)
    
    return overpriced, underpriced


def calculate_error_metrics(comparison_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate error metrics for the pricing model.
    
    Args:
        comparison_df: DataFrame with comparison results
        
    Returns:
        Dictionary with error metrics
    """
    # Filter for valid price differences (non-NaN)
    valid_df = comparison_df.dropna(subset=['price_diff'])
    
    if len(valid_df) == 0:
        return {'MAE': float('nan'), 'RMSE': float('nan'), 'MAPE': float('nan')}
    
    # Calculate metrics
    mae = np.abs(valid_df['price_diff']).mean()
    rmse = np.sqrt((valid_df['price_diff'] ** 2).mean())
    
    # MAPE calculation (avoiding division by zero)
    valid_prices = valid_df[valid_df['lastPrice'] > 0]
    mape = np.abs(valid_prices['price_diff_pct']).mean() if len(valid_prices) > 0 else float('nan')
    
    return {
        'MAE': mae,  # Mean Absolute Error
        'RMSE': rmse,  # Root Mean Square Error
        'MAPE': mape,  # Mean Absolute Percentage Error
    }


def analyze_by_moneyness(comparison_df: pd.DataFrame, 
                        price_col: str = 'underlying_price') -> Dict[str, Dict[str, float]]:
    """
    Analyze pricing differences by option moneyness.
    
    Args:
        comparison_df: DataFrame with comparison results
        price_col: Column name for underlying price
        
    Returns:
        Dictionary with analysis by moneyness category
    """
    # Make a copy and ensure required columns exist
    df = comparison_df.copy()
    
    if price_col not in df.columns or 'strike' not in df.columns:
        return {}
    
    # Calculate moneyness (S/K ratio)
    df['moneyness'] = df[price_col] / df['strike']
    
    # Categorize options by moneyness
    df['moneyness_category'] = 'ATM'  # At-the-money
    
    # For call options
    call_mask = df['option_type'] == 'call'
    df.loc[call_mask & (df['moneyness'] < 0.95), 'moneyness_category'] = 'OTM'  # Out-of-the-money
    df.loc[call_mask & (df['moneyness'] > 1.05), 'moneyness_category'] = 'ITM'  # In-the-money
    
    # For put options
    put_mask = df['option_type'] == 'put'
    df.loc[put_mask & (df['moneyness'] > 1.05), 'moneyness_category'] = 'OTM'
    df.loc[put_mask & (df['moneyness'] < 0.95), 'moneyness_category'] = 'ITM'
    
    # Analyze by moneyness category
    result = {}
    for category in ['ITM', 'ATM', 'OTM']:
        category_df = df[df['moneyness_category'] == category]
        if len(category_df) > 0:
            metrics = calculate_error_metrics(category_df)
            avg_diff = category_df['price_diff'].mean()
            avg_diff_pct = category_df['price_diff_pct'].dropna().mean()
            count = len(category_df)
            
            result[category] = {
                'count': count,
                'avg_diff': avg_diff,
                'avg_diff_pct': avg_diff_pct,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE']
            }
    
    return result


def analyze_by_expiration(comparison_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze pricing differences by expiration timeframe.
    
    Args:
        comparison_df: DataFrame with comparison results
        
    Returns:
        Dictionary with analysis by expiration category
    """
    # Make a copy and ensure required columns exist
    df = comparison_df.copy()
    
    if 'daysToExpiry' not in df.columns and 'T' not in df.columns:
        return {}
    
    # Use daysToExpiry if available, otherwise calculate from T
    if 'daysToExpiry' not in df.columns:
        df['daysToExpiry'] = df['T'] * 365
    
    # Categorize options by expiration timeframe
    df['expiry_category'] = 'Medium-term'  # 30-90 days
    df.loc[df['daysToExpiry'] < 30, 'expiry_category'] = 'Short-term'  # < 30 days
    df.loc[df['daysToExpiry'] > 90, 'expiry_category'] = 'Long-term'  # > 90 days
    
    # Analyze by expiration category
    result = {}
    for category in ['Short-term', 'Medium-term', 'Long-term']:
        category_df = df[df['expiry_category'] == category]
        if len(category_df) > 0:
            metrics = calculate_error_metrics(category_df)
            avg_diff = category_df['price_diff'].mean()
            avg_diff_pct = category_df['price_diff_pct'].dropna().mean()
            count = len(category_df)
            
            result[category] = {
                'count': count,
                'avg_diff': avg_diff,
                'avg_diff_pct': avg_diff_pct,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE']
            }
    
    return result


def generate_comparison_report(comparison_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive comparison report.
    
    Args:
        comparison_df: DataFrame with comparison results
        
    Returns:
        Dictionary with report data
    """
    # Overall error metrics
    overall_metrics = calculate_error_metrics(comparison_df)
    
    # Analysis by moneyness
    moneyness_analysis = analyze_by_moneyness(comparison_df)
    
    # Analysis by expiration
    expiry_analysis = analyze_by_expiration(comparison_df)
    
    # Identify mispriced options
    overpriced, underpriced = identify_mispriced_options(comparison_df)
    
    # Calculate implied volatility skew
    vol_skew = {}
    if 'calculated_implied_volatility' in comparison_df.columns:
        # Group by strike and option type, then calculate mean IV
        iv_by_strike = comparison_df.groupby(['strike', 'option_type'])['calculated_implied_volatility'].mean()
        if not iv_by_strike.empty:
            vol_skew = {
                'strikes': iv_by_strike.index.get_level_values('strike').unique().tolist(),
                'call_ivs': [iv_by_strike.get((k, 'call'), float('nan')) for k in iv_by_strike.index.get_level_values('strike').unique()],
                'put_ivs': [iv_by_strike.get((k, 'put'), float('nan')) for k in iv_by_strike.index.get_level_values('strike').unique()]
            }
    
    # Compile the report
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'options_count': len(comparison_df),
        'overall_metrics': overall_metrics,
        'moneyness_analysis': moneyness_analysis,
        'expiry_analysis': expiry_analysis,
        'overpriced_count': len(overpriced),
        'underpriced_count': len(underpriced),
        'top_overpriced': overpriced.head(5).to_dict('records') if not overpriced.empty else [],
        'top_underpriced': underpriced.head(5).to_dict('records') if not underpriced.empty else [],
        'volatility_skew': vol_skew
    }
    
    return report


def save_comparison_results(comparison_df: pd.DataFrame, output_dir: str = None):
    """
    Save comparison results to CSV files.
    
    Args:
        comparison_df: DataFrame with comparison results
        output_dir: Directory to save the results
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = "data/processed"
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a timestamp for the filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the full comparison DataFrame
    comparison_df.to_csv(f"{output_dir}/comparison_results_{timestamp}.csv", index=False)
    
    # Identify mispriced options and save separately
    overpriced, underpriced = identify_mispriced_options(comparison_df)
    
    if not overpriced.empty:
        overpriced.to_csv(f"{output_dir}/overpriced_options_{timestamp}.csv", index=False)
    
    if not underpriced.empty:
        underpriced.to_csv(f"{output_dir}/underpriced_options_{timestamp}.csv", index=False)
    
    print(f"Comparison results saved to {output_dir}")


if __name__ == "__main__":
    """Test the comparison module with sample data."""
    
    # Create a sample options dataset for testing
    import numpy as np
    
    # Current date and underlying price
    current_price = 100.0
    
    # Generate sample options data
    strikes = np.arange(80, 121, 5)  # Strike prices from 80 to 120 in steps of 5
    days_to_expiry = [30, 60, 90]  # Different expiration dates
    
    # Create a list to hold the sample data
    sample_data = []
    
    for days in days_to_expiry:
        T = days / 365  # Convert to years
        for K in strikes:
            # Theoretical values
            call_price = bs_price(current_price, K, T, 0.05, 0.2, 'call')
            put_price = bs_price(current_price, K, T, 0.05, 0.2, 'put')
            
            # Add some random noise to create "market" prices
            call_market = call_price * (1 + np.random.uniform(-0.1, 0.1))
            put_market = put_price * (1 + np.random.uniform(-0.1, 0.1))
            
            # Add call option
            sample_data.append({
                'option_type': 'call',
                'strike': K,
                'daysToExpiry': days,
                'T': T,
                'underlying_price': current_price,
                'lastPrice': call_market,
                'historical_volatility': 0.2,
                'risk_free_rate': 0.05
            })
            
            # Add put option
            sample_data.append({
                'option_type': 'put',
                'strike': K,
                'daysToExpiry': days,
                'T': T,
                'underlying_price': current_price,
                'lastPrice': put_market,
                'historical_volatility': 0.2,
                'risk_free_rate': 0.05
            })
    
    # Convert to DataFrame
    sample_df = pd.DataFrame(sample_data)
    
    print("Sample Options Data:")
    print(sample_df.head())
    
    # Test the comparison analysis
    print("\nCalculating price differences...")
    comparison_results = calculate_price_differences(sample_df)
    
    print("\nCalculating implied volatility...")
    comparison_results = calculate_implied_volatility_surface(comparison_results)
    
    print("\nIdentifying mispriced options...")
    overpriced, underpriced = identify_mispriced_options(comparison_results)
    
    print(f"\nFound {len(overpriced)} overpriced and {len(underpriced)} underpriced options")
    
    if not overpriced.empty:
        print("\nTop 3 Overpriced Options:")
        print(overpriced[['option_type', 'strike', 'daysToExpiry', 'lastPrice', 'theoretical_price', 'price_diff_pct']].head(3))
    
    if not underpriced.empty:
        print("\nTop 3 Underpriced Options:")
        print(underpriced[['option_type', 'strike', 'daysToExpiry', 'lastPrice', 'theoretical_price', 'price_diff_pct']].head(3))
    
    # Calculate error metrics
    metrics = calculate_error_metrics(comparison_results)
    print("\nError Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # Analyze by moneyness
    moneyness_analysis = analyze_by_moneyness(comparison_results)
    print("\nAnalysis by Moneyness:")
    for category, values in moneyness_analysis.items():
        print(f"{category}: Count={values['count']}, Avg Diff=${values['avg_diff']:.4f}, Avg Diff%={values['avg_diff_pct']:.2f}%")
    
    # Generate a comprehensive report
    report = generate_comparison_report(comparison_results)
    print("\nReport generated successfully")
    
    # Optionally save the results
    # save_comparison_results(comparison_results, "data/processed")