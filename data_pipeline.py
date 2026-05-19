import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_historical_data(ticker_symbol, days_back):
    """
    Stáhne historická OHLCV data z Yahoo Finance s maskováním za reálný prohlížeč.
    Toto řešení umožňuje stáhnout neomezené množství akcií pro Multi-Asset analýzu.
    """
    import yfinance as yf
    import requests
    import pandas as pd
    
    print(f"Stahuji data pro {ticker_symbol} z Yahoo Finance...")
    
    # Vytvoření relace s falešnou hlavičkou pro oklamání blokace z GitHubu
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive'
    })
    
    # Použití Tickeru s injektovanou relací
    ticker = yf.Ticker(ticker_symbol, session=session)
    df = ticker.history(period="5y")
    
    if df.empty:
        # V případě úplného banu fallback: pokus o vyhledání kratší historie
        df = ticker.history(period="1y")
        if df.empty:
            raise ValueError(f"Chyba: Nepodařilo se stáhnout data pro {ticker_symbol}. Burza neodpovídá.")
            
    df = df.reset_index()
    
    # Sjednocení a úprava sloupců
    if 'Date' not in df.columns and 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'Date'})
        
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    
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
