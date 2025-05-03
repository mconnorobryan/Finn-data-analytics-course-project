# src/analytics/greeks.py
"""
Functions to calculate option Greeks (Delta, Gamma, Theta, Vega, Rho).

This module provides analytical formulas for calculating the sensitivity
of option prices to various parameters in the Black-Scholes model.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Any, Literal, Union, Optional

import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Import Black-Scholes functions
from src.models.black_scholes import d1, d2, bs_price


def delta(S: float, K: float, T: float, r: float, sigma: float, 
         option_type: Literal['call', 'put'] = 'call') -> float:
    """
    Calculate the Delta of an option.
    Delta measures the rate of change of option price with respect to changes in underlying price.
    
    Args:
        S: Current stock price
        K: Option strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility of the underlying asset
        option_type: Type of option ('call' or 'put')
        
    Returns:
        Delta value
    """
    # Handle edge cases
    if T <= 0:
        # At expiration
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    
    if sigma <= 0:
        # No volatility (deterministic case)
        if option_type == 'call':
            return 1.0 if S > K * np.exp(-r * T) else 0.0
        else:
            return -1.0 if S < K * np.exp(-r * T) else 0.0
    
    d1_val = d1(S, K, T, r, sigma)
    
    if option_type == 'call':
        return norm.cdf(d1_val)
    else:  # put
        return norm.cdf(d1_val) - 1  # or equivalently: -norm.cdf(-d1_val)


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the Gamma of an option.
    Gamma measures the rate of change of Delta with respect to changes in underlying price.
    
    Args:
        S: Current stock price
        K: Option strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility of the underlying asset
        
    Returns:
        Gamma value (same for both call and put)
    """
    # Handle edge cases
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1_val = d1(S, K, T, r, sigma)
    
    # Formula for Gamma (same for calls and puts)
    return norm.pdf(d1_val) / (S * sigma * np.sqrt(T))


def theta(S: float, K: float, T: float, r: float, sigma: float, 
         option_type: Literal['call', 'put'] = 'call') -> float:
    """
    Calculate the Theta of an option.
    Theta measures the rate of change of option price with respect to passage of time.
    
    Args:
        S: Current stock price
        K: Option strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility of the underlying asset
        option_type: Type of option ('call' or 'put')
        
    Returns:
        Theta value (per year, annualized)
    """
    # Handle edge cases
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    
    # Common term for both calls and puts
    common_term = -(S * norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T))
    
    if option_type == 'call':
        return common_term - r * K * np.exp(-r * T) * norm.cdf(d2_val)
    else:  # put
        return common_term + r * K * np.exp(-r * T) * norm.cdf(-d2_val)


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the Vega of an option.
    Vega measures the rate of change of option price with respect to changes in volatility.
    
    Args:
        S: Current stock price
        K: Option strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility of the underlying asset
        
    Returns:
        Vega value (same for both call and put, per 1% change in volatility)
    """
    # Handle edge cases
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1_val = d1(S, K, T, r, sigma)
    
    # Formula for Vega (same for calls and puts)
    # Divided by 100 to express per 1% change in volatility
    return S * norm.pdf(d1_val) * np.sqrt(T) / 100


def rho(S: float, K: float, T: float, r: float, sigma: float, 
       option_type: Literal['call', 'put'] = 'call') -> float:
    """
    Calculate the Rho of an option.
    Rho measures the rate of change of option price with respect to changes in risk-free rate.
    
    Args:
        S: Current stock price
        K: Option strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility of the underlying asset
        option_type: Type of option ('call' or 'put')
        
    Returns:
        Rho value (per 1% change in interest rate)
    """
    # Handle edge cases
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d2_val = d2(S, K, T, r, sigma)
    
    # Formula for Rho, different for calls and puts
    # Divided by 100 to express per 1% change in interest rate
    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2_val) / 100
    else:  # put
        return -K * T * np.exp(-r * T) * norm.cdf(-d2_val) / 100


def calculate_all_greeks(params: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate all Greeks for an option based on input parameters.
    
    Args:
        params: Dictionary containing the following keys:
            - S or underlying_price: Current stock price
            - K or strike: Option strike price
            - T or time_to_expiry: Time to expiration in years
            - r or risk_free_rate: Risk-free interest rate
            - sigma or volatility: Volatility of the underlying asset
            - option_type: Type of option ('call' or 'put')
    
    Returns:
        Dictionary containing all calculated Greeks
    """
    # Extract parameters with support for different key names
    S = params.get('S', params.get('underlying_price', 0))
    K = params.get('K', params.get('strike', 0))
    T = params.get('T', params.get('time_to_expiry', 0))
    r = params.get('r', params.get('risk_free_rate', 0.05))
    sigma = params.get('sigma', params.get('volatility', 0))
    option_type = params.get('option_type', 'call')
    
    # Calculate theoretical price
    price = bs_price(S, K, T, r, sigma, option_type)
    
    # Calculate all Greeks
    greeks = {
        'price': price,
        'delta': delta(S, K, T, r, sigma, option_type),
        'gamma': gamma(S, K, T, r, sigma),
        'theta': theta(S, K, T, r, sigma, option_type),
        'theta_daily': theta(S, K, T, r, sigma, option_type) / 365,  # Daily theta
        'vega': vega(S, K, T, r, sigma),
        'rho': rho(S, K, T, r, sigma, option_type)
    }
    
    return greeks


def create_sensitivity_table(base_params: Dict[str, Any], 
                            param_name: str, 
                            param_range: list) -> Dict[str, list]:
    """
    Create a sensitivity table by varying one parameter across a range.
    
    Args:
        base_params: Base parameters for option pricing
        param_name: Name of the parameter to vary
        param_range: List of values for the parameter
        
    Returns:
        Dictionary with parameter values and corresponding Greeks
    """
    results = {
        'param_values': param_range,
        'price': [],
        'delta': [],
        'gamma': [],
        'theta': [],
        'vega': [],
        'rho': []
    }
    
    for value in param_range:
        # Create a copy of the base parameters and update the specified parameter
        params = base_params.copy()
        params[param_name] = value
        
        # Calculate Greeks with the updated parameters
        greeks = calculate_all_greeks(params)
        
        # Store results
        results['price'].append(greeks['price'])
        results['delta'].append(greeks['delta'])
        results['gamma'].append(greeks['gamma'])
        results['theta'].append(greeks['theta'])
        results['vega'].append(greeks['vega'])
        results['rho'].append(greeks['rho'])
        
    return results


if __name__ == "__main__":
    """Test the Greeks calculations with example values."""
    
    # Example values for testing
    S = 100.0  # Current stock price
    K = 100.0  # Strike price (at-the-money)
    T = 1.0    # Time to expiry (1 year)
    r = 0.05   # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    
    # Calculate Greeks for call option
    call_greeks = {
        'delta': delta(S, K, T, r, sigma, 'call'),
        'gamma': gamma(S, K, T, r, sigma),
        'theta': theta(S, K, T, r, sigma, 'call'),
        'vega': vega(S, K, T, r, sigma),
        'rho': rho(S, K, T, r, sigma, 'call')
    }
    
    # Calculate Greeks for put option
    put_greeks = {
        'delta': delta(S, K, T, r, sigma, 'put'),
        'gamma': gamma(S, K, T, r, sigma),
        'theta': theta(S, K, T, r, sigma, 'put'),
        'vega': vega(S, K, T, r, sigma),
        'rho': rho(S, K, T, r, sigma, 'put')
    }
    
    print("Greeks Calculation Test")
    print("======================")
    print(f"Stock Price (S): ${S:.2f}")
    print(f"Strike Price (K): ${K:.2f}")
    print(f"Time to Expiry (T): {T:.2f} years")
    print(f"Risk-free Rate (r): {r:.2%}")
    print(f"Volatility (σ): {sigma:.2%}")
    
    print("\nCall Option Greeks:")
    print(f"Delta: {call_greeks['delta']:.4f}")
    print(f"Gamma: {call_greeks['gamma']:.6f}")
    print(f"Theta: {call_greeks['theta']:.4f} (per year)")
    print(f"Theta: {call_greeks['theta']/365:.6f} (per day)")
    print(f"Vega: {call_greeks['vega']:.4f} (per 1% change in volatility)")
    print(f"Rho: {call_greeks['rho']:.4f} (per 1% change in interest rate)")
    
    print("\nPut Option Greeks:")
    print(f"Delta: {put_greeks['delta']:.4f}")
    print(f"Gamma: {put_greeks['gamma']:.6f}")
    print(f"Theta: {put_greeks['theta']:.4f} (per year)")
    print(f"Theta: {put_greeks['theta']/365:.6f} (per day)")
    print(f"Vega: {put_greeks['vega']:.4f} (per 1% change in volatility)")
    print(f"Rho: {put_greeks['rho']:.4f} (per 1% change in interest rate)")
    
    # Test using the all Greeks function
    print("\nUsing calculate_all_greeks function:")
    params = {
        'underlying_price': 100.0,
        'strike': 100.0,
        'time_to_expiry': 1.0,
        'risk_free_rate': 0.05,
        'volatility': 0.2,
        'option_type': 'call'
    }
    
    all_greeks = calculate_all_greeks(params)
    for greek, value in all_greeks.items():
        print(f"{greek.capitalize()}: {value:.6f}")
        
    # Test creating a sensitivity table by varying the stock price
    print("\nSensitivity to Stock Price:")
    stock_prices = [90, 95, 100, 105, 110]
    sensitivity = create_sensitivity_table(params, 'underlying_price', stock_prices)
    
    print(f"{'Stock Price':<15}{'Delta':<15}{'Gamma':<15}{'Theta (day)':<15}")
    for i, price in enumerate(sensitivity['param_values']):
        print(f"${price:<14.2f}{sensitivity['delta'][i]:<14.4f}{sensitivity['gamma'][i]:<14.6f}{sensitivity['theta'][i]/365:<14.6f}")