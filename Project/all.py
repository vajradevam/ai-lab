import os
import re
import glob
import time
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from bs4 import BeautifulSoup

import google.generativeai as genai

from Modelyaga import big_model_baba

load_dotenv()
KEY = os.environ['GEMINI']

# Configure Gemini API
genai.configure(api_key=KEY)  # Replace with your actual API key

# Set up the model
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

def cleanup():
    # Define the directory
    output_dir = "TICKER_DATA"

    # Find all CSV files in the directory
    files = glob.glob(os.path.join(output_dir, "*.csv"))

    # Delete each file
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

    print("Cleanup completed. All CSV files deleted.")

def cleanup_csv_data(file):
    # Read the CSV file
    df = pd.read_csv(file)

    # List of columns to set to NaN (all except Symbol, Company Name, Sector)
    cols_to_nan = ['Previous Price', 'Predicted Price', 'Change ($)', 'Change (%)', 
                'New/Old Ratio', 'MSE', 'R2', 'Sentiment Score', 'Sentiment Validity']

    # Set these columns to NaN
    df[cols_to_nan] = np.nan

    # Save or display the result
    df.to_csv(file, index=False)

def get_ticker_data():
    # Create a directory to save data
    output_dir = "TICKER_DATA"
    os.makedirs(output_dir, exist_ok=True)

    # Load tickers from CSV (Assuming 'tickers.csv' has a column 'Symbol')
    tickers_df = pd.read_csv("tickers.csv")
    tickers = tickers_df["Symbol"].tolist()

    # Define start and end dates (past 1 year)
    end_date = pd.to_datetime("today").strftime('%Y-%m-%d')
    start_date = (pd.to_datetime("today") - pd.DateOffset(years=1)).strftime('%Y-%m-%d')

    # Fetch historical stock prices for each ticker and save separately
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(start=start_date, end=end_date, interval="1d")
            if not history.empty:
                history.reset_index(inplace=True)
                history["Symbol"] = ticker  # Add ticker column
                
                # Save as CSV
                file_path = os.path.join(output_dir, f"{ticker}.csv")
                history.to_csv(file_path, index=False)
                print(f"Saved: {file_path}")
            else:
                print(f"No data for {ticker}")

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    print(f"All stock data saved in '{output_dir}' directory.")

def predict_and_update():
    # Read the tickers information
    tickers_info = pd.read_csv('tickers_ml.csv')

    # Initialize lists to store all data
    results = []

    ticker_dir = "./TICKER_DATA/"
    ticker_files = os.listdir("./TICKER_DATA/")

    # Process each file
    for ticker_file in ticker_files:
        ticker = ticker_file.split('.')[0]
        file_path = os.path.join(ticker_dir, ticker_file)
        
        # Get predictions and stats
        pred, stat = big_model_baba(file_path)
        
        # Read last actual price from CSV
        df = pd.read_csv(file_path)
        prev_price = df['Close'].iloc[-1]
        
        # Calculate metrics
        change_dollar = pred - prev_price
        change_percent = (change_dollar / prev_price) * 100
        new_old_ratio = pred / prev_price
        
        # Get company info from tickers_info
        company_info = tickers_info[tickers_info['Symbol'] == ticker].iloc[0]
        
        # Create result dictionary
        result = {
            'Symbol': ticker,
            'Company Name': company_info['Company Name'],
            'Sector': company_info['Sector'],
            'Previous Price': prev_price,
            'Predicted Price': pred,
            'Change ($)': change_dollar,
            'Change (%)': change_percent,
            'New/Old Ratio': new_old_ratio,
            'MSE': stat['mse'],
            'R2': stat['r2']
            # Add any other stats from stat dictionary here
        }
        
        results.append(result)
        
        # Print results in one row (optional)
        # print(f"{ticker:<6} | {prev_price:10.2f} | {pred:10.2f} | {change_dollar:8.2f} | {change_percent:8.2f}% | {new_old_ratio:8.4f} | {stat['mse']:8.2f} | {stat['r2']:8.4f}")

        # Create DataFrame from results
        results_df = pd.DataFrame(results)

        # Reorder columns if needed
        columns_order = ['Symbol', 'Company Name', 'Sector', 'Previous Price', 'Predicted Price', 
                        'Change ($)', 'Change (%)', 'New/Old Ratio', 'MSE', 'R2']
        results_df = results_df[columns_order]

        # Save to CSV if desired
        results_df.to_csv('tickers_ml.csv', index=False)

def get_news(ticker):
    ticker = yf.Ticker(ticker)

    news = ticker.news

    news_text = ""

    for item in news:
        news_text += f"Title: {item["content"]["title"]}\n" + f"Summary: {item["content"]["summary"]}\n"

    return news_text

def analyze_news_sentiment(news_text: str) -> List[Dict]:
    """
    Analyzes financial news text and returns sentiment analysis for mentioned tickers
    
    Args:
        news_text: The financial news text to analyze
        
    Returns:
        List of dictionaries with ticker, sentiment score, and sentiment validity
        Format: [{"ticker": str, "sentiment score": float, "sentiment validity": float}]
    """
    prompt = f"""Analyze this financial news text and perform the following tasks:
    
    1. Identify all stock tickers mentioned (e.g., AAPL, MSFT, GOOGL)
    2. For each ticker, determine the sentiment (positive/negative/neutral)
    3. Assign a sentiment score between 0 (most negative) and 1 (most positive)
    4. Assign a sentiment validity score between 0 (low confidence) and 1 (high confidence)
       based on how clearly the sentiment is expressed for that ticker
    
    Return ONLY a JSON-formatted list of dictionaries with these keys:
    - "ticker" (the stock symbol)
    - "sentiment score" (0-1)
    - "sentiment validity" (0-1)
    
    News text to analyze:
    {news_text}
    """
    
    try:
        response = model.generate_content(prompt)
        
        # Extract JSON from the response
        json_str = response.text.strip().replace('```json', '').replace('```', '').strip()
        
        # Parse the JSON string into Python objects
        import json
        result = json.loads(json_str)
        
        # Validate the structure
        if not isinstance(result, list):
            raise ValueError("Response is not a list")
            
        for item in result:
            if not all(key in item for key in ["ticker", "sentiment score", "sentiment validity"]):
                raise ValueError("Missing required keys in response")
                
        return result
        
    except Exception as e:
        print(f"Error analyzing news: {e}")
        return []


def update_tickers_with_sentiment(csv_path: str, sentiment_results: List[Dict]) -> pd.DataFrame:
    """
    Updates the tickers CSV with sentiment analysis results
    
    Args:
        csv_path: Path to the tickers CSV file
        sentiment_results: List of sentiment analysis dictionaries from Gemini
        
    Returns:
        Updated pandas DataFrame
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Initialize new columns if they don't exist
        if 'Sentiment Score' not in df.columns:
            df['Sentiment Score'] = None
        if 'Sentiment Validity' not in df.columns:
            df['Sentiment Validity'] = None
        
        # Create a mapping from sentiment results for quick lookup
        sentiment_map = {
            item['ticker']: {
                'score': item['sentiment score'],
                'validity': item['sentiment validity']
            }
            for item in sentiment_results
        }
        
        # Update the DataFrame
        for index, row in df.iterrows():
            ticker = row['Symbol']
            if ticker in sentiment_map:
                df.at[index, 'Sentiment Score'] = sentiment_map[ticker]['score']
                df.at[index, 'Sentiment Validity'] = sentiment_map[ticker]['validity']
        
        # Optionally save back to CSV
        df.to_csv(f"tickers_news.csv", index=False)
        
        return df
        
    except Exception as e:
        print(f"Error updating tickers CSV: {e}")
        return pd.DataFrame()

def process_tickers_news():
    for ticker in os.listdir("./TICKER_DATA/"):
        news_text = get_news(ticker[:-4])
        sentiment_results = analyze_news_sentiment(news_text)
        update_tickers_with_sentiment("./tickers_news.csv", sentiment_results)
        time.sleep(2)

cleanup()
cleanup_csv_data("./tickers.csv")
cleanup_csv_data("./tickers_ml.csv")
cleanup_csv_data("./tickers_news.csv")

get_ticker_data()

import multiprocessing

def main():
    # Create two separate processes
    process1 = multiprocessing.Process(target=predict_and_update)
    process2 = multiprocessing.Process(target=process_tickers_news)
    
    # Start both processes
    process1.start()
    process2.start()
    
    # Wait for both processes to complete
    process1.join()
    process2.join()

main()

def merge_stock_data(prediction_file, sentiment_file, output_file):
    # Read the files
    predictions = pd.read_csv(prediction_file)
    sentiments = pd.read_csv(sentiment_file)
    
    # Identify columns to add from sentiment file (excluding overlapping columns)
    sentiment_cols_to_add = [col for col in sentiments.columns 
                           if col not in predictions.columns]
    
    # Merge while keeping all prediction columns and only adding sentiment columns
    merged_data = predictions.merge(
        sentiments[['Symbol', 'Company Name', 'Sector'] + sentiment_cols_to_add],
        on=['Symbol', 'Company Name', 'Sector'],
        how='left'
    )
    
    # Save the merged data
    merged_data.to_csv(output_file, index=False)
    print(f"Merged data saved to {output_file}")
    
    return merged_data

prediction_csv = "tickers_ml.csv"
sentiment_csv = "tickers_news.csv"
output_csv = "tickers.csv"

result = merge_stock_data(prediction_csv, sentiment_csv, output_csv)

input_csv = "tickers.csv"
df = pd.read_csv(input_csv)
df['Relative_MSE'] = (df['MSE'] / (df['Previous Price'] ** 2)) * 100
df.to_csv(input_csv, index=False)

ddff = pd.read_csv("./tickers.csv")
ddff

import pandas as pd

# Define weights and normalization constant
alpha = 1.0  # weight for price momentum
beta = 5.0   # weight for sentiment term
gamma = 10.0  # weight for new/old ratio term
delta = 1.0  # weight for R2
epsilon = 1.0  # weight for MSE term

# Read CSV file
df = pd.read_csv('tickers.csv')

# Compute the price momentum: predicted percentage change in price
df['Price_Momentum'] = (df['Predicted Price'] - df['Previous Price']) / df['Previous Price']

# Compute the sentiment term: product of sentiment score and its validity
df['Sentiment_Term'] = df['Sentiment Score'] * df['Sentiment Validity']

# Compute the adjusted new/old ratio term: centered around 0
df['Relative_Ratio'] = df['New/Old Ratio'] - 1

# Compute the composite signal using the proposed formula
df['Signal'] = (
    alpha * df['Price_Momentum'] +
    beta * df['Sentiment_Term'] +
    gamma * df['Relative_Ratio'] +
    delta * df['R2'] -
    epsilon * (df['Relative_MSE'])
)

df = df.sort_values(by='Signal', ascending=False)

# Display the updated DataFrame
df.to_csv("final.csv", index=False)