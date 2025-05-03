# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import json

# Import your existing analysis modules
from src.data.loader import get_stock_data, get_options_chain, get_risk_free_rate
from src.data.processor import calculate_historical_volatility, clean_options_data, prepare_model_inputs
from src.models.black_scholes import bs_price, calculate_option_price, implied_volatility
from src.analytics.comparison import (
    calculate_price_differences, 
    calculate_implied_volatility_surface,
    identify_mispriced_options,
    generate_comparison_report
)
from src.visualization.option_charts import (
    plot_option_price_curve,
    plot_price_comparison,
    plot_implied_volatility_smile,
    plot_volatility_surface,
    plot_price_differences_heatmap
)

# Set page config
st.set_page_config(
    page_title="Options Pricing Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Options Pricing and Greeks Analysis")
st.markdown(
    """
    This dashboard provides interactive analysis of options pricing using the Black-Scholes model.
    Compare theoretical prices with market prices, identify potential mispricing, and analyze option Greeks.
    """
)

# Sidebar for inputs
st.sidebar.header("Configuration")

# Stock ticker input
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")

# Date range selection
today = datetime.now()
default_start_date = today - timedelta(days=365)
default_end_date = today

start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date", today)

# Convert dates to string format
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Analysis parameters
st.sidebar.subheader("Analysis Parameters")
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
volatility_source = st.sidebar.radio("Volatility Source", ["Historical", "Custom"])

if volatility_source == "Custom":
    custom_volatility = st.sidebar.slider("Custom Volatility (%)", 5.0, 100.0, 20.0) / 100
else:
    custom_volatility = None

# Run analysis button
run_analysis = st.sidebar.button("Run Analysis")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Stock Data", "Options Analysis", "Mispricing", "Report"])

# Function to run the analysis
def perform_analysis(ticker, start_date, end_date, risk_free_rate, custom_volatility=None):
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Get data
    status_text.text("Loading stock data...")
    stock_data = get_stock_data(ticker, start_date, end_date)
    if stock_data.empty:
        st.error(f"Error: Could not fetch stock data for {ticker}")
        return None, None, None, None, None
    
    progress_bar.progress(20)
    status_text.text("Loading options chain data...")
    options_data = get_options_chain(ticker)
    if not options_data:
        st.error(f"Error: Could not fetch options data for {ticker}")
        return None, None, None, None, None
    
    # Step 2: Process data
    progress_bar.progress(40)
    status_text.text("Processing data...")
    processed_stock = calculate_historical_volatility(stock_data)
    cleaned_options = clean_options_data(options_data)
    
    # Use custom volatility if provided
    if custom_volatility is not None:
        current_volatility = custom_volatility
    else:
        # Use historical volatility if available, otherwise default to 20%
        if not processed_stock['volatility'].dropna().empty:
            current_volatility = processed_stock['volatility'].dropna().iloc[-1]
        else:
            current_volatility = 0.2
    
    prepared_stock, prepared_options = prepare_model_inputs(processed_stock, cleaned_options, risk_free_rate)
    current_price = float(prepared_stock['Close'].iloc[-1])
    
    # Step 3: Calculate theoretical prices and Greeks
    progress_bar.progress(60)
    status_text.text("Calculating theoretical prices...")
    
    # Combine options data for analysis
    expiry_dates = sorted(prepared_options.keys())
    analyzed_expiries = expiry_dates[:min(3, len(expiry_dates))]  # Analyze first 3 expirations
    
    combined_options = pd.DataFrame()
    for expiry in analyzed_expiries:
        df = prepared_options[expiry].copy()
        df['expiry_date'] = expiry
        combined_options = pd.concat([combined_options, df])
    
    # Calculate theoretical prices and differences
    comparison_results = calculate_price_differences(combined_options)
    comparison_results = calculate_implied_volatility_surface(comparison_results)
    
    # Identify potential mispricing
    progress_bar.progress(80)
    status_text.text("Identifying mispriced options...")
    overpriced, underpriced = identify_mispriced_options(comparison_results)
    
    # Generate report
    progress_bar.progress(90)
    status_text.text("Generating report...")
    report = generate_comparison_report(comparison_results)
    
    # Analysis complete
    progress_bar.progress(100)
    status_text.text("Analysis complete!")
    
    return processed_stock, comparison_results, overpriced, underpriced, report

# Main content
if run_analysis:
    with st.spinner("Running options pricing analysis..."):
        processed_stock, comparison_results, overpriced, underpriced, report = perform_analysis(
            ticker, start_date_str, end_date_str, risk_free_rate, custom_volatility
        )
    
    if processed_stock is not None:
        # Display results in each tab
        
        # Tab 1: Stock Data
        with tab1:
            st.subheader(f"{ticker} Stock Price History")
            
            # Plot stock price history
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(processed_stock.index, processed_stock['Close'], 'g-', label='Close Price')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.set_title(f"{ticker} Stock Price History")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            
            # Display volatility
            st.subheader("Volatility Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                if not processed_stock['volatility'].dropna().empty:
                    current_vol = processed_stock['volatility'].dropna().iloc[-1]
                    st.metric(
                        "Current Historical Volatility (30-day)", 
                        f"{current_vol:.2%}"
                    )
                else:
                    st.metric(
                        "Current Historical Volatility (30-day)", 
                        "N/A", 
                        help="Not enough data to calculate historical volatility"
                    )
            
            with col2:
                current_price = float(processed_stock['Close'].iloc[-1])
                st.metric(
                    "Current Stock Price", 
                    f"${current_price:.2f}"
                )
            
            # Plot historical volatility if available
            if 'volatility' in processed_stock.columns and not processed_stock['volatility'].dropna().empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(processed_stock.index, processed_stock['volatility'], 'b-', label='30-Day Historical Volatility')
                ax.set_xlabel('Date')
                ax.set_ylabel('Volatility')
                ax.set_title(f"{ticker} Historical Volatility")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
            
            # Display recent stock data
            st.subheader("Recent Stock Data")
            st.dataframe(processed_stock.tail(10))
        
        # Tab 2: Options Analysis
        with tab2:
            st.subheader("Options Chain Analysis")
            
            if comparison_results is not None and not comparison_results.empty:
                # Filter for specific expiration date
                expiry_dates = sorted(comparison_results['expiry_date'].unique())
                selected_expiry = st.selectbox("Select Expiration Date", expiry_dates)
                
                expiry_data = comparison_results[comparison_results['expiry_date'] == selected_expiry]
                days_to_expiry = expiry_data['daysToExpiry'].iloc[0]
                
                st.markdown(f"**Analyzing options expiring on {selected_expiry} ({days_to_expiry} days to expiry)**")
                
                # Filter for option type
                option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)
                option_data = expiry_data[expiry_data['option_type'] == option_type.lower()]
                
                # Display price comparison
                if not option_data.empty:
                    st.subheader(f"{option_type} Option Price Analysis")
                    
                    # Create price comparison plot
                    fig = plot_price_comparison(
                        option_data['strike'].values,
                        option_data['theoretical_price'].values,
                        option_data['lastPrice'].values,
                        option_type.lower(),
                        f"{option_type} Option Price Comparison ({days_to_expiry} days to expiry)"
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Display options data in table
                    st.subheader(f"{option_type} Options Data")
                    display_cols = ['strike', 'lastPrice', 'theoretical_price', 'price_diff', 
                                    'price_diff_pct', 'calculated_implied_volatility', 'daysToExpiry']
                    
                    # Rename columns for display
                    display_data = option_data[display_cols].copy()
                    display_data.columns = ['Strike', 'Market Price', 'Theoretical Price', 'Price Difference', 
                                           'Price Diff (%)', 'Implied Volatility', 'Days to Expiry']
                    
                    # Format display data
                    display_data['Implied Volatility'] = display_data['Implied Volatility'].apply(
                        lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A"
                    )
                    display_data['Price Diff (%)'] = display_data['Price Diff (%)'].apply(
                        lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A"
                    )
                    
                    st.dataframe(display_data)
                    
                    # Implied volatility smile
                    iv_data = option_data.dropna(subset=['calculated_implied_volatility'])
                    if not iv_data.empty:
                        st.subheader("Implied Volatility Smile")
                        fig = plot_implied_volatility_smile(
                            iv_data['strike'].values,
                            iv_data['calculated_implied_volatility'].values,
                            current_price,
                            option_type.lower(),
                            days_to_expiry
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                else:
                    st.warning(f"No {option_type.lower()} options data available for {selected_expiry}")
            else:
                st.warning("No options data available for analysis.")
        
        # Tab 3: Mispricing
        with tab3:
            st.subheader("Potentially Mispriced Options")
            
            if overpriced is not None and underpriced is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Overpriced Options", len(overpriced))
                
                with col2:
                    st.metric("Underpriced Options", len(underpriced))
                
                # Display overpriced options
                if not overpriced.empty:
                    st.subheader("Top Overpriced Options")
                    display_cols = ['option_type', 'strike', 'expiry_date', 'lastPrice', 
                                   'theoretical_price', 'price_diff', 'price_diff_pct']
                    
                    # Rename columns for display
                    display_data = overpriced[display_cols].head(10).copy()
                    display_data.columns = ['Type', 'Strike', 'Expiry', 'Market Price', 
                                           'Theoretical Price', 'Price Diff', 'Price Diff (%)']
                    
                    # Format data
                    display_data['Type'] = display_data['Type'].str.upper()
                    display_data['Price Diff (%)'] = display_data['Price Diff (%)'].apply(
                        lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A"
                    )
                    
                    st.dataframe(display_data)
                else:
                    st.info("No overpriced options found.")
                
                # Display underpriced options
                if not underpriced.empty:
                    st.subheader("Top Underpriced Options")
                    display_cols = ['option_type', 'strike', 'expiry_date', 'lastPrice', 
                                   'theoretical_price', 'price_diff', 'price_diff_pct']
                    
                    # Rename columns for display
                    display_data = underpriced[display_cols].head(10).copy()
                    display_data.columns = ['Type', 'Strike', 'Expiry', 'Market Price', 
                                           'Theoretical Price', 'Price Diff', 'Price Diff (%)']
                    
                    # Format data
                    display_data['Type'] = display_data['Type'].str.upper()
                    display_data['Price Diff (%)'] = display_data['Price Diff (%)'].apply(
                        lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A"
                    )
                    
                    st.dataframe(display_data)
                else:
                    st.info("No underpriced options found.")
                
                # Price differences heatmap
                if comparison_results is not None and not comparison_results.empty:
                    st.subheader("Price Differences Heatmap")
                    fig = plot_price_differences_heatmap(
                        comparison_results,
                        'price_diff_pct',
                        f"{ticker} Option Price Differences (%)"
                    )
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.warning("No mispricing analysis available.")
        
        # Tab 4: Report
        with tab4:
            st.subheader("Analysis Report")
            
            if report is not None:
                # Display overall metrics
                st.subheader("Overall Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Mean Absolute Error", 
                        f"${report['overall_metrics']['MAE']:.4f}"
                    )
                
                with col2:
                    st.metric(
                        "Root Mean Square Error", 
                        f"${report['overall_metrics']['RMSE']:.4f}"
                    )
                
                with col3:
                    st.metric(
                        "Mean Absolute % Error", 
                        f"{report['overall_metrics']['MAPE']:.2f}%"
                    )
                
                # Analysis by moneyness
                st.subheader("Analysis by Moneyness")
                moneyness_data = []
                for category, metrics in report['moneyness_analysis'].items():
                    moneyness_data.append({
                        'Category': category,
                        'Count': metrics['count'],
                        'Avg. Diff': f"${metrics['avg_diff']:.4f}",
                        'Avg. Diff (%)': f"{metrics['avg_diff_pct']:.2f}%",
                        'MAE': f"${metrics['MAE']:.4f}",
                        'RMSE': f"${metrics['RMSE']:.4f}"
                    })
                
                st.table(pd.DataFrame(moneyness_data))
                
                # Analysis by expiration
                st.subheader("Analysis by Expiration")
                expiry_data = []
                for category, metrics in report['expiry_analysis'].items():
                    expiry_data.append({
                        'Category': category,
                        'Count': metrics['count'],
                        'Avg. Diff': f"${metrics['avg_diff']:.4f}",
                        'Avg. Diff (%)': f"{metrics['avg_diff_pct']:.2f}%",
                        'MAE': f"${metrics['MAE']:.4f}",
                        'RMSE': f"${metrics['RMSE']:.4f}"
                    })
                
                st.table(pd.DataFrame(expiry_data))
                
                # Summary and conclusion
                st.subheader("Key Findings")
                
                # Find the category with largest price difference
                max_diff_category = max(report['moneyness_analysis'].items(), 
                                       key=lambda x: abs(x[1]['avg_diff_pct']))
                
                st.markdown(f"1. **{max_diff_category[0]}** options show the largest pricing discrepancy "
                           f"({max_diff_category[1]['avg_diff_pct']:.2f}%).")
                
                # Find if there's a volatility skew
                if 'volatility_skew' in report and report['volatility_skew']:
                    st.markdown("2. There is evidence of **volatility skew** in the options chain, with out-of-the-money "
                              "options typically having higher implied volatility than at-the-money options.")
                else:
                    st.markdown("2. The analysis did not find significant volatility skew in the options chain.")
                
                # Download full report as JSON
                if st.button("Download Full Report (JSON)"):
                    report_json = json.dumps(report, indent=2, default=str)
                    st.download_button(
                        "Click to Download",
                        report_json,
                        file_name=f"{ticker}_options_analysis_report.json",
                        mime="application/json"
                    )
            else:
                st.warning("No analysis report available.")
else:
    # Initial screen when no analysis has been run yet
    st.info("Configure parameters in the sidebar and click 'Run Analysis' to start.")
    
    # Display sample image or instructions
    st.markdown("""
    ### How to use this dashboard:
    
    1. Enter a stock ticker symbol in the sidebar (e.g., AAPL, MSFT, GOOGL)
    2. Set the date range for historical data analysis
    3. Adjust analysis parameters if needed
    4. Click "Run Analysis" to generate results
    5. Explore different tabs to view stock data, options analysis, mispricing, and the report
    
    This dashboard uses the Black-Scholes model to calculate theoretical option prices and 
    compares them with market prices to identify potential mispricing opportunities.
    """)

# Add footer
st.markdown("---")
st.markdown("Options Pricing and Greeks Analysis Dashboard | Created with Streamlit")