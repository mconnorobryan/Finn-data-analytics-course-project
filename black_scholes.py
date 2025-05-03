# src/models/black_scholes.py
"""
Implementation of the Black-Scholes option pricing model.

This module provides functions to calculate theoretical option prices
using the Black-Scholes-Merton model, including support for both
European call and put options.
"""

import numpy as np
from scipy.stats import norm
from typing import Literal, Union, Dict, Any, Optional

import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the d1 term in the Black-Scholes formula.
    
    Args:
        S: Current stock price
        K: Option strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal form, e.g., 0.05 for 5%)
        sigma: Volatility of the underlying asset (decimal form)
        
    Returns:
        d1 term value
    """
    # Handle edge cases
    if sigma <= 0 or T <= 0:
        return float('nan')
        
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the d2 term in the Black-Scholes formula.
    
    Args:
        S: Current stock price
        K: Option strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal form)
        sigma: Volatility of the underlying asset (decimal form)
        
    Returns:
        d2 term value
    """
    # Handle edge cases
    if sigma <= 0 or T <= 0:
        return float('nan')
        
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes price for a European call option.
    
    Args:
        S: Current stock price
        K: Option strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal form)
        sigma: Volatility of the underlying asset (decimal form)
        
    Returns:
        Theoretical price of the call option
    """
    # Handle edge cases
    if T <= 0:
        return max(0, S - K)  # Intrinsic value at expiration
    
    if sigma <= 0:
        if S > K:
            return S - K * np.exp(-r * T)  # Deterministic case
        else:
            return 0  # Out of the money
    
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    
    # Black-Scholes formula for call option
    call_price = S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)
    
    return call_price


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes price for a European put option.
    
    Args:
        S: Current stock price
        K: Option strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal form)
        sigma: Volatility of the underlying asset (decimal form)
        
    Returns:
        Theoretical price of the put option
    """
    # Handle edge cases
    if T <= 0:
        return max(0, K - S)  # Intrinsic value at expiration
    
    if sigma <= 0:
        if S < K:
            return K * np.exp(-r * T) - S  # Deterministic case
        else:
            return 0  # Out of the money
    
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    
    # Black-Scholes formula for put option
    put_price = K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)
    
    return put_price


def bs_price(S: float, K: float, T: float, r: float, sigma: float, 
            option_type: Literal['call', 'put'] = 'call') -> float:
    """
    Calculate Black-Scholes option price for either call or put.
    
    Args:
        S: Current stock price
        K: Option strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal form)
        sigma: Volatility of the underlying asset (decimal form)
        option_type: Type of option ('call' or 'put')
        
    Returns:
        Theoretical option price according to Black-Scholes model
    """
    if option_type.lower() == 'call':
        return bs_call_price(S, K, T, r, sigma)
    elif option_type.lower() == 'put':
        return bs_put_price(S, K, T, r, sigma)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

def calculate_option_price(params: Dict[str, Any]) -> float:
    """
    Helper function to calculate option price from a parameter dictionary.
    
    Args:
        params: Dictionary containing the following keys:
            - S or underlying_price: Current stock price
            - K or strike: Option strike price
            - T or time_to_expiry: Time to expiration in years
            - r or risk_free_rate: Risk-free interest rate
            - sigma or volatility: Volatility of the underlying asset
            - option_type: Type of option ('call' or 'put')
    
    Returns:
        Option price according to Black-Scholes model
    """
    # Extract parameters with support for different key names
    S = params.get('S', params.get('underlying_price', 0))
    K = params.get('K', params.get('strike', 0))
    T = params.get('T', params.get('time_to_expiry', 0))
    r = params.get('r', params.get('risk_free_rate', 0.05))
    sigma = params.get('sigma', params.get('volatility', 0))
    option_type = params.get('option_type', 'call')
    
    return bs_price(S, K, T, r, sigma, option_type)

def implied_volatility(price: float, S: float, K: float, T: float, r: float, 
                      option_type: Literal['call', 'put'] = 'call', 
                      precision: float = 0.00001,
                      max_iterations: int = 100) -> Optional[float]:
    """
    Calculate implied volatility using the Newton-Raphson method.
    
    Args:
        price: Market price of the option
        S: Current stock price
        K: Option strike price
        T: Time to expiration in years
        r: Risk-free interest rate (decimal form)
        option_type: Type of option ('call' or 'put')
        precision: Desired precision for the implied volatility
        max_iterations: Maximum number of iterations for the algorithm
        
    Returns:
        Implied volatility as a decimal, or None if algorithm doesn't converge
    """
    # Initial guess
    sigma = 0.2  # Start with 20% volatility
    
    for i in range(max_iterations):
        # Calculate option price with current sigma
        price_diff = bs_price(S, K, T, r, sigma, option_type) - price
        
        if abs(price_diff) < precision:
            return sigma
        
        # Calculate vega (sensitivity of price to volatility