import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_historical_data(ticker_symbol, days_back):
    """
    Stáhne historická OHLCV data přes spolehlivé Alpha Vantage API (obchází zákazy na cloudu).
    """
    import os
    import requests
    import pandas as pd
    
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Chybí ALPHA_VANTAGE_API_KEY v Secrets!")
        
    print(f"Stahuji data pro {ticker_symbol} z Alpha Vantage...")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker_symbol}&outputsize=compact&apikey={api_key}"
    
    response = requests.get(url)
    data = response.json()
    
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Chyba Alpha Vantage API (pravděpodobně vyčerpán limit?): {data}")
        
    ts = data["Time Series (Daily)"]
    
    # Prevod z JSON na DataFrame
    df = pd.DataFrame.from_dict(ts, orient='index')
    df.index.name = 'Date'
    df = df.reset_index()
    
    # Prejmenovani sloupcu a prevod na cisla
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume"
    })
    
    # Data přichází seřazena od nejnovějšího k nejstaršímu, musíme to otočit
    df = df.sort_values('Date', ascending=True)
    
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col])
        
    df['Date'] = pd.to_datetime(df['Date'])
    
    if len(df) > days_back:
        df = df.tail(days_back).copy()
        
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df

def calculate_technical_indicators(df):
    """
    Vypočítá základní technické indikátory pro modelování.
    """
    df = df.copy()
    # Jednoduché klouzavé průměry
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    
    # Volatilita
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=10).std()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df = df.fillna(0) # Pro zjednodušení nahradíme NaN nuly na začátku
    return df

def simulate_rag_sentiment(dates):
    """
    Mock funkce, která simuluje výstup z RAG/LLM architektury analyzující 10-K a zprávy.
    V produkci by tato funkce dotazovala Qdrant DB a Gemini/GPT-4o API.
    Vrací skóre od -1.0 do 1.0.
    """
    # Pro účely historického backtestu vygenerujeme syntetický sentiment,
    # který má mírnou korelaci s budoucím vývojem nebo je to jen random walk
    np.random.seed(42)
    sentiment_scores = np.random.normal(loc=0.05, scale=0.3, size=len(dates))
    sentiment_scores = np.clip(sentiment_scores, -1.0, 1.0)
    return sentiment_scores

def prepare_dataset(ticker="SPY", days_back=365, for_training=True):
    """
    Hlavní funkce pipeline pro přípravu finálního DataFrame s Features.
    """
    df = get_historical_data(ticker, days_back)
    df = calculate_technical_indicators(df)
    df['Sentiment_Score'] = simulate_rag_sentiment(df['Date'])
    
    # Definice cílové proměnné (Target) pro Supervised Learning: 
    # Bude zítřejší cena vyšší než dnešní? (1 = Ano, 0 = Ne)
    df['Target_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Odstranění posledního řádku (nemá Target) POUZE při tréninku
    if for_training:
        df = df.iloc[:-1]
    
    return df

if __name__ == "__main__":
    data = prepare_dataset()
    print("Ukázka datové sady s fúzí Quant a RAG Sentimentu:")
    print(data.tail())
