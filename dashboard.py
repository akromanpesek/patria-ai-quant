import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json
from datetime import datetime, timedelta
from data_pipeline import prepare_dataset
import pickle

st.set_page_config(page_title="AI Quant Multi-Asset Systém", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1 {color: #00ffcc;}
    .metric-container {background-color: #1e2532; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #333; color: white !important;}
    .metric-container h3, .metric-container h2 {color: white !important;}
    </style>
    """, unsafe_allow_html=True)

st.title("📈 AI Quant - Osobní Správce Portfolia")

# Načtení portfolia
portfolio_path = "portfolio.json"
try:
    with open(portfolio_path, "r", encoding="utf-8") as f:
        portfolio = json.load(f)
    assets = portfolio.get("assets", [])
    cash = portfolio.get("cash", 0)
except Exception as e:
    st.error("Chyba při čtení portfolia: " + str(e))
    assets = []
    cash = 0

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="metric-container"><h3>Dostupná Hotovost</h3><h2>' + f"${cash:,.2f}" + '</h2></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-container"><h3>Sledovaných firem</h3><h2>' + str(len(assets)) + '</h2></div>', unsafe_allow_html=True)

st.write("---")
st.subheader("Živá Analýza Portfolia a Trhu")

if st.button("🚀 SPUSTIT AI ANALÝZU NA VŠECHNY AKCIE"):
    model_path = os.path.join('trained_models', 'best_deep_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        st.success(f"Mozek úspěšně připojen: {model.name.split('_Gen')[0]}")
        features = ['SMA_10', 'SMA_30', 'Volatility', 'RSI', 'Sentiment_Score']
        
        results = []
        progress_bar = st.progress(0)
        
        for i, asset in enumerate(assets):
            ticker = asset["ticker"]
            shares = asset.get("shares", 0)
            buy_price = asset.get("buy_price", 0)
            
            try:
                df = prepare_dataset(ticker=ticker, days_back=60, for_training=False)
                last_row = df.iloc[-1:]
                current_price = last_row['Close'].iloc[0]
                prediction = model.predict(last_row[features])[0]
                
                # Zisk/Ztráta
                if shares > 0:
                    profit = (current_price - buy_price) * shares
                    profit_perc = ((current_price / buy_price) - 1) * 100
                    profit_str = f"${profit:.2f} ({profit_perc:.1f}%)"
                else:
                    profit_str = "-"
                    
                # Signál
                if prediction == 1:
                    sig_text = "🟢 DRŽET" if shares > 0 else "🟢 KOUPIT"
                else:
                    sig_text = "🔴 PRODAT" if shares > 0 else "🟡 IGNOROVAT"
                    
                results.append({
                    "Akcie": ticker,
                    "Aktuální Cena": f"${current_price:.2f}",
                    "V mém portfoliu": f"{shares} ks",
                    "Nákupka": f"${buy_price:.2f}" if shares > 0 else "-",
                    "Zisk/Ztráta": profit_str,
                    "AI Pokyn pro zítřek": sig_text
                })
            except Exception as e:
                results.append({"Akcie": ticker, "AI Pokyn pro zítřek": "❌ Chyba dat"})
                
            progress_bar.progress((i + 1) / len(assets))
            
        st.table(pd.DataFrame(results))
        st.info("💡 **Jak upravit portfolio:** Na svém GitHubu klikněte na soubor `portfolio.json`, dejte ikonku tužky, přepište počty nakoupených akcií a dejte Commit. Dashboard a Telegram se automaticky přizpůsobí!")
    else:
        st.error("Chyba: Model nebyl nalezen.")
