# main_analysis.py
"""
Main script for running the Options Pricing and Greeks Analysis.

This script ties together all the components of the project to perform
a comprehensive analysis of option pricing and Greeks.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Import our modules
from src.data.loader import get_stock_data, get_options_chain, get_risk_free_rate
from src.data.processor import calculate_historical_volatility, clean_options_data, prepare_model_inputs
from src.models.black_scholes import bs_price, calculate_option_price, implied_volatility
from src.analytics.greeks import calculate_all_greeks
from src.analytics.comparison import (
    calculate_price_differences, 
    calculate_implied_volatility_surface,
    identify_mispriced_options,
    generate_comparison_report,
    save_comparison_results
)
from src.visualization.option_charts import (
    plot_option_price_curve,
    plot_price_comparison,
    plot_implied_volatility_smile,
    plot_volatility_surface,
    plot_price_differences_heatmap
)
from src.visualization.greek_charts import (
    plot_all_greeks,
    plot_greek_heatmap,
    generate_all_greek_charts
)


def run_analysis(ticker, start_date, end_date, output_dir):
    """
    Run the complete options pricing and Greeks analysis for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for historical data
        end_date: End date for historical data
        output_dir: Directory for saving outputs
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    print(f"Running Options Pricing and Greeks Analysis for {ticker}")
    print(f"Data period: {start_date} to {end_date}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Get data
    print("\nStep 1: Gathering Data")
    print("----------------------")
    
    print("Fetching stock data...")
    stock_data = get_stock_data(ticker, start_date, end_date)
    if stock_data.empty:
        print("Error: Could not fetch stock data. Exiting.")
        return
    
    print(f"Successfully downloaded {len(stock_data)} rows of stock data")
    
    print("\nFetching options chain data...")
    options_data = get_options_chain(ticker)
    if not options_data:
        print("Error: Could not fetch options data. Exiting.")
        return
    
    print(f"Successfully downloaded options data for {len(options_data)} expiration dates")
    
    # Step 2: Process data
    print("\nStep 2: Processing Data")
    print("----------------------")
    
    print("Calculating historical volatility...")
    processed_stock = calculate_historical_volatility(stock_data)
    
    print("Processing options data...")
    cleaned_options = clean_options_data(options_data)
    
    # Get risk-free rate
    risk_free_rate = get_risk_free_rate()
    print(f"Using risk-free rate: {risk_free_rate:.2%}")
    
    print("\nPreparing model inputs...")
    prepared_stock, prepared_options = prepare_model_inputs(processed_stock, cleaned_options, risk_free_rate)
    
    current_price = float(prepared_stock['Close'].iloc[-1])
    if prepared_stock['volatility'].dropna().empty:
        current_volatility = 0.2  # Default 20% volatility
        print(f"Warning: No volatility data available, using default value of 20%")
    else:
        current_volatility = prepared_stock['volatility'].dropna().iloc[-1]
    
    print(f"Current stock price: ${current_price:.2f}")
    print(f"Historical volatility (30-day): {current_volatility:.2%}")
    
    # Step 3: Calculate theoretical prices and Greeks
    print("\nStep 3: Calculating Theoretical Prices and Greeks")
    print("-----------------------------------------------")
    
    # Find the nearest expiration date for detailed analysis
    if not prepared_options:
        print("No options data available for analysis.")
        return
    
    # Sort expiration dates and select a few for analysis
    expiry_dates = sorted(prepared_options.keys())
    analyzed_expiries = []
    
    # Select short, medium, and long-term expiries if available
    if len(expiry_dates) >= 3:
        analyzed_expiries = [expiry_dates[0], expiry_dates[len(expiry_dates)//2], expiry_dates[-1]]
    else:
        analyzed_expiries = expiry_dates
    
    # Combine options data for the selected expiries
    combined_options = pd.DataFrame()
    for expiry in analyzed_expiries:
        df = prepared_options[expiry].copy()
        df['expiry_date'] = expiry
        combined_options = pd.concat([combined_options, df])
    
    print(f"Analyzing {len(combined_options)} options across {len(analyzed_expiries)} expiration dates")
    
    # Calculate theoretical prices and Greeks
    print("Calculating theoretical prices...")
    comparison_results = calculate_price_differences(combined_options)
    
    print("Calculating implied volatility...")
    comparison_results = calculate_implied_volatility_surface(comparison_results)
    
    # Identify potential mispricing
    print("\nIdentifying potentially mispriced options...")
    overpriced, underpriced = identify_mispriced_options(comparison_results)
    
    print(f"Found {len(overpriced)} potentially overpriced options")
    print(f"Found {len(underpriced)} potentially underpriced options")
    
    # Generate comprehensive report
    print("\nGenerating analysis report...")
    report = generate_comparison_report(comparison_results)
    
    # Save results
    print("\nSaving analysis results...")
    save_comparison_results(comparison_results, output_dir)
    
    # Save report as JSON
    import json
    report_file = os.path.join(output_dir, "analysis_report.json")
    def json_serial(obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.to_list()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Add any other special cases here
        raise TypeError(f"Type {type(obj)} not serializable")

    # Save report as JSON
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=json_serial)
    
    # Step 4: Create visualizations
    print("\nStep 4: Creating Visualizations")
    print("----------------------------")
    
    # Group data by expiration date
    grouped_by_expiry = comparison_results.groupby('expiry_date')
    
    for expiry, group in grouped_by_expiry:
        expiry_name = expiry.replace('-', '')
        days_to_expiry = group['daysToExpiry'].iloc[0]
        
        print(f"\nGenerating charts for expiration date: {expiry} ({days_to_expiry} days)")
        
        # Create directory for this expiry
        expiry_dir = os.path.join(charts_dir, expiry_name)
        os.makedirs(expiry_dir, exist_ok=True)
        
        # Filter call and put options
        calls = group[group['option_type'] == 'call'].sort_values('strike')
        puts = group[group['option_type'] == 'put'].sort_values('strike')
        
        if not calls.empty:
            # Option price curves
            print("  Creating call option price curve...")
            fig = plot_option_price_curve(
                calls['strike'].values,
                calls['theoretical_price'].values,
                None,
                current_price,
                f"Call Option Prices ({days_to_expiry} days to expiry)",
                os.path.join(expiry_dir, "call_price_curve.png")
            )
            plt.close(fig)
            
            # Theoretical vs market price comparison
            print("  Creating call price comparison chart...")
            fig = plot_price_comparison(
                calls['strike'].values,
                calls['theoretical_price'].values,
                calls['lastPrice'].values,
                'call',
                f"Call Option Price Comparison ({days_to_expiry} days to expiry)",
                os.path.join(expiry_dir, "call_price_comparison.png")
            )
            plt.close(fig)
            
           
        
        if not puts.empty:
            # Option price curves - FIX HERE
            print("  Creating put option price curve...")
            fig = plot_option_price_curve(
                puts['strike'].values,
                puts['theoretical_price'].values,
                None,
                current_price,
                f"Put Option Prices ({days_to_expiry} days to expiry)",
                os.path.join(expiry_dir, "put_price_curve.png")
            )
            plt.close(fig)  # FIXED: Changed from "for fig in call_figs.values(): plt.close(fig)"
            
            # Theoretical vs market price comparison
            print("  Creating put price comparison chart...")
            fig = plot_price_comparison(
                puts['strike'].values,
                puts['theoretical_price'].values,
                puts['lastPrice'].values,
                'put',
                f"Put Option Price Comparison ({days_to_expiry} days to expiry)",
                os.path.join(expiry_dir, "put_price_comparison.png")
            )
            plt.close(fig)
            
            
        
        # Create implied volatility smile if we have both calls and puts
        if not calls.empty and not puts.empty:
            print("  Creating implied volatility smile...")
            # Calls
            call_iv = calls.dropna(subset=['calculated_implied_volatility'])
            if not call_iv.empty:
                fig = plot_implied_volatility_smile(
                    call_iv['strike'].values,
                    call_iv['calculated_implied_volatility'].values,
                    current_price,
                    'call',
                    days_to_expiry,
                    f"Call Option Implied Volatility Smile ({days_to_expiry} days to expiry)",
                    os.path.join(expiry_dir, "call_iv_smile.png")
                )
                plt.close(fig)
            
            # Puts
            put_iv = puts.dropna(subset=['calculated_implied_volatility'])
            if not put_iv.empty:
                fig = plot_implied_volatility_smile(
                    put_iv['strike'].values,
                    put_iv['calculated_implied_volatility'].values,
                    current_price,
                    'put',
                    days_to_expiry,
                    f"Put Option Implied Volatility Smile ({days_to_expiry} days to expiry)",
                    os.path.join(expiry_dir, "put_iv_smile.png")
                )
                plt.close(fig)
    
    # Create combined visualizations across all expirations
    print("\nGenerating combined visualizations...")
    
    # Price differences heatmap
    print("  Creating price differences heatmap...")
    fig = plot_price_differences_heatmap(
        comparison_results,
        'price_diff_pct',
        f"{ticker} Option Price Differences (%)",
        os.path.join(charts_dir, "price_diff_heatmap.png")
    )
    plt.close(fig)
    
    # Implied volatility surface if we have enough data
    iv_data = comparison_results.dropna(subset=['calculated_implied_volatility'])
    if len(iv_data['strike'].unique()) > 3 and len(iv_data['daysToExpiry'].unique()) > 2:
        print("  Creating implied volatility surface...")
        fig = plot_volatility_surface(
            iv_data,
            f"{ticker} Implied Volatility Surface",
            os.path.join(charts_dir, "iv_surface.png")
        )
        plt.close(fig)
    
    # Historical volatility
    print("  Creating historical volatility chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(processed_stock.index, processed_stock['volatility'], 'b-', label='30-Day Historical Volatility')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')
    ax.set_title(f"{ticker} Historical Volatility")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig(os.path.join(charts_dir, "historical_volatility.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Stock price history
    print("  Creating stock price history chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(processed_stock.index, processed_stock['Close'], 'g-', label='Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title(f"{ticker} Stock Price History")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig(os.path.join(charts_dir, "stock_price_history.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Step 5: Generate summary report
    print("\nStep 5: Generating Summary Report")
    print("------------------------------")
    
    # Create a text summary report
    report_path = os.path.join(output_dir, "summary_report.txt")
    
    with open(report_path, 'w') as f:
        f.write(f"OPTIONS PRICING AND GREEKS ANALYSIS FOR {ticker}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("ANALYSIS OVERVIEW\n")
        f.write("-" * 30 + "\n")
        f.write(f"Date of Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Stock: {ticker}\n")
        f.write(f"Current Price: ${current_price:.2f}\n")
        f.write(f"Historical Volatility (30-day): {current_volatility:.2%}\n")
        f.write(f"Risk-free Rate: {risk_free_rate:.2%}\n\n")
        
        f.write("ANALYZED EXPIRATION DATES\n")
        f.write("-" * 30 + "\n")
        for expiry in analyzed_expiries:
            days = prepared_options[expiry]['daysToExpiry'].iloc[0]
            option_count = len(prepared_options[expiry])
            f.write(f"- {expiry} ({days} days to expiry): {option_count} options\n")
        f.write("\n")
        
        f.write("PRICING ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Options Analyzed: {len(comparison_results)}\n")
        f.write(f"Mean Absolute Error (MAE): ${report['overall_metrics']['MAE']:.4f}\n")
        f.write(f"Root Mean Square Error (RMSE): ${report['overall_metrics']['RMSE']:.4f}\n")
        f.write(f"Mean Absolute Percentage Error (MAPE): {report['overall_metrics']['MAPE']:.2f}%\n\n")
        
        f.write("MISPRICING ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Potentially Overpriced Options: {len(overpriced)}\n")
        f.write(f"Potentially Underpriced Options: {len(underpriced)}\n\n")
        
        if not overpriced.empty:
            f.write("Top 5 Overpriced Options:\n")
            for i, row in overpriced.head(5).iterrows():
                f.write(f"  {row['option_type'].upper()} ${row['strike']:.2f} ({row['expiry_date']}): ")
                f.write(f"Market ${row['lastPrice']:.2f} vs. Theoretical ${row['theoretical_price']:.2f} ")
                f.write(f"({row['price_diff_pct']:.2f}% diff)\n")
            f.write("\n")
        
        if not underpriced.empty:
            f.write("Top 5 Underpriced Options:\n")
            for i, row in underpriced.head(5).iterrows():
                f.write(f"  {row['option_type'].upper()} ${row['strike']:.2f} ({row['expiry_date']}): ")
                f.write(f"Market ${row['lastPrice']:.2f} vs. Theoretical ${row['theoretical_price']:.2f} ")
                f.write(f"({row['price_diff_pct']:.2f}% diff)\n")
            f.write("\n")
        
        f.write("ANALYSIS BY MONEYNESS\n")
        f.write("-" * 30 + "\n")
        for category, metrics in report['moneyness_analysis'].items():
            f.write(f"{category} Options (n={metrics['count']}):\n")
            f.write(f"  Avg. Price Diff: ${metrics['avg_diff']:.4f} ({metrics['avg_diff_pct']:.2f}%)\n")
            f.write(f"  MAE: ${metrics['MAE']:.4f}\n")
            f.write(f"  RMSE: ${metrics['RMSE']:.4f}\n")
        f.write("\n")
        
        f.write("ANALYSIS BY EXPIRATION\n")
        f.write("-" * 30 + "\n")
        for category, metrics in report['expiry_analysis'].items():
            f.write(f"{category} Options (n={metrics['count']}):\n")
            f.write(f"  Avg. Price Diff: ${metrics['avg_diff']:.4f} ({metrics['avg_diff_pct']:.2f}%)\n")
            f.write(f"  MAE: ${metrics['MAE']:.4f}\n")
            f.write(f"  RMSE: ${metrics['RMSE']:.4f}\n")
        f.write("\n")
        
        f.write("GENERATED VISUALIZATIONS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Output Directory: {charts_dir}\n\n")
        f.write("Main Charts:\n")
        f.write("- Historical Volatility\n")
        f.write("- Stock Price History\n")
        f.write("- Price Differences Heatmap\n")
        if len(iv_data['strike'].unique()) > 3 and len(iv_data['daysToExpiry'].unique()) > 2:
            f.write("- Implied Volatility Surface\n")
        f.write("\n")
        
        f.write("Charts by Expiration Date:\n")
        for expiry in analyzed_expiries:
            days = prepared_options[expiry]['daysToExpiry'].iloc[0]
            expiry_name = expiry.replace('-', '')
            f.write(f"- {expiry} ({days} days):\n")
            f.write("  - Option Price Curves\n")
            f.write("  - Price Comparisons\n")
            f.write("  - Greek Curves\n")
            f.write("  - Implied Volatility Smiles\n")
        f.write("\n")
        
        f.write("CONCLUSION\n")
        f.write("-" * 30 + "\n")
        f.write("The Black-Scholes model provides a theoretical framework for options pricing, but real market\n")
        f.write("prices often deviate due to factors not captured by the model such as skew, kurtosis,\n")
        f.write("liquidity considerations, and market sentiment.\n\n")
        
        f.write("Key findings from this analysis:\n")
        # Find the category with largest price difference
        max_diff_category = max(report['moneyness_analysis'].items(), 
                               key=lambda x: abs(x[1]['avg_diff_pct']))
        f.write(f"1. {max_diff_category[0]} options show the largest pricing discrepancy ")
        f.write(f"({max_diff_category[1]['avg_diff_pct']:.2f}%).\n")
        
        # Find if there's a volatility skew
        if 'volatility_skew' in report and report['volatility_skew']:
            f.write("2. There is evidence of volatility skew in the options chain, with out-of-the-money\n")
            f.write("   options typically having higher implied volatility than at-the-money options.\n")
        else:
            f.write("2. The analysis did not find significant volatility skew in the options chain.\n")
        
        # Add observation about term structure if we have multiple expiries
        if len(analyzed_expiries) > 1:
            f.write("3. The term structure of implied volatility indicates ")
            if len(iv_data['daysToExpiry'].unique()) > 2:
                # Check if longer-term options have higher IV
                short_term = iv_data[iv_data['daysToExpiry'] <= 30]['calculated_implied_volatility'].mean()
                long_term = iv_data[iv_data['daysToExpiry'] >= 90]['calculated_implied_volatility'].mean()
                if long_term > short_term:
                    f.write("higher volatility expectations for longer-term options.\n")
                else:
                    f.write("higher volatility expectations for shorter-term options.\n")
            else:
                f.write("varying volatility expectations across different expirations.\n")
    
    print(f"\nAnalysis complete! Summary report saved to {report_path}")
    print(f"All visualization charts saved to {charts_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options Pricing and Greeks Analysis')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, default='2024-01-01', help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2025-04-30', help='End date for historical data (YYYY-MM-DD)')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory for saving outputs')
    
    args = parser.parse_args()
    
    run_analysis(args.ticker, args.start_date, args.end_date, args.output_dir)