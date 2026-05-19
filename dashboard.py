import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import os
import json
from datetime import datetime, timedelta
from data_pipeline import prepare_dataset
import pickle

st.set_page_config(page_title="AI Quant Trading Systém", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1 {color: #00ffcc;}
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 24px;
        background-color: #00ffcc;
        color: #000;
        border-radius: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #00ccaa;
        color: white;
    }
    .metric-container {
        background-color: #1e2532;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #333;
        color: white !important;
    }
    .metric-container h3, .metric-container h2 {
        color: white !important;
    }
    .date-header {
        font-size: 18px;
        color: #888;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 AI Quant - Kvantový Broker Dashboard")
st.markdown(f'<div class="date-header">📅 <b>Dnešní datum:</b> {datetime.now().strftime("%d. %m. %Y")} &nbsp;|&nbsp; ⏱️ <b>Den našeho obchodování:</b> 2</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🚀 Hlavní Strategie (S&P 500)", "💼 Osobní Multi-Asset Správce"])

# --- TAB 1: Původní S&P 500 ---
with tab1:
    if 'sim_run' not in st.session_state:
        st.session_state.sim_run = False
    if 'final_equity' not in st.session_state:
        st.session_state.final_equity = 10000.0
    if 'winning_model' not in st.session_state:
        st.session_state.winning_model = ""
    if 'signal' not in st.session_state:
        st.session_state.signal = ""
    if 'equity_curve' not in st.session_state:
        st.session_state.equity_curve = []
    if 'dates' not in st.session_state:
        st.session_state.dates = []

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-container"><h3>Aktuální Hotovost</h3><h2>' + f"${st.session_state.final_equity:,.2f}" + '</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container"><h3>Nejlepší Model</h3><h2>' + (st.session_state.winning_model if st.session_state.winning_model else "N/A") + '</h2></div>', unsafe_allow_html=True)
    with col3:
        signal_color = "#ff4444" if "DRŽET HOTOVOST" in st.session_state.signal else "#00ffcc" if "KOUPIT" in st.session_state.signal else "#ffffff"
        st.markdown(f'<div class="metric-container"><h3>Dnešní Signál (SPY)</h3><h2 style="color:{signal_color} !important;">' + (st.session_state.signal if st.session_state.signal else "Čeká na spuštění") + '</h2></div>', unsafe_allow_html=True)

    st.write("---")
    
    if st.button("🧠 NAČÍST OSTRÝ MODEL Z NOČNÍHO TRÉNINKU (SPY)"):
        model_path = os.path.join('trained_models', 'best_deep_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            short_name = loaded_model.name.split("_Gen")[0] + " (Ostrý Cloud Model)"
            st.session_state.winning_model = short_name
            st.session_state.deep_success = True
            
            try:
                df = prepare_dataset(ticker="SPY", days_back=60, for_training=False)
                features = ['SMA_10', 'SMA_30', 'Volatility', 'RSI', 'Sentiment_Score']
                last_row = df.iloc[-1:]
                prediction = loaded_model.predict(last_row[features])[0]
                
                if prediction == 1:
                    st.session_state.signal = "KOUPIT SPY<br><span style='font-size:16px;'>(ISIN: US78462F1030)</span>"
                else:
                    st.session_state.signal = "DRŽET HOTOVOST<br><span style='font-size:16px;'>(Nekupovat SPY)</span>"
                    
                st.session_state.sim_run = True
                dates = pd.date_range(end=datetime.now(), periods=18, freq='D')
                st.session_state.dates = dates.tolist()
                st.session_state.equity_curve = [10000.00] * 18
                st.session_state.final_equity = 10000.00
                
            except Exception as e:
                st.error(f"❌ Nelze stáhnout živá data: {e}")
                st.session_state.signal = "CHYBA DAT"
                st.session_state.sim_run = False
        else:
            st.error("Chyba: Model nebyl na disku nalezen.")

    if st.session_state.sim_run:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.dates, y=st.session_state.equity_curve, mode='lines+markers', name='Hodnota účtu', line=dict(color='#00ffcc', width=3)))
        fig.update_layout(plot_bgcolor='#0e1117', paper_bgcolor='#0e1117', font=dict(color='white'), margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Historie mých pokynů pro vás")
        clean_signal = st.session_state.signal.split("<br>")[0] if st.session_state.signal else "DRŽET HOTOVOST"
        trade_data = {
            "Datum": [(datetime.now() - timedelta(days=4-i)).strftime("%Y-%m-%d") for i in range(5)],
            "Den obchodování": ["Den -4", "Den -3", "Den -2", "Den 1 (Včera)", "Dnes (Den 2)"],
            "Můj Pokyn (Signál)": ["KOUPIT", "DRŽET", "PRODAT", "DRŽET HOTOVOST", clean_signal],
            "Provedli jsme nákup?": ["Simulace", "Simulace", "Simulace", "Čekal se start", "Záleží na vás"]
        }
        st.table(pd.DataFrame(trade_data))

# --- TAB 2: Nový Multi-Asset ---
with tab2:
    st.subheader("Živá Analýza Osobního Portfolia")
    portfolio_path = "portfolio.json"
    try:
        with open(portfolio_path, "r", encoding="utf-8") as f:
            portfolio = json.load(f)
        assets = portfolio.get("assets", [])
    except:
        assets = []
        
    st.markdown(f"Nalezeno **{len(assets)}** osobních akcií v paměti robota.")
    
    if st.button("🚀 SPUSTIT AI ANALÝZU MOJICH AKCIÍ"):
        model_path = os.path.join('trained_models', 'best_deep_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
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
                    
                    if prediction == 1:
                        sig_text = "🟢 DRŽET (Růst)" if shares > 0 else "🟢 KOUPIT"
                    else:
                        sig_text = "🔴 PRODAT (Klesá)" if shares > 0 else "🟡 IGNOROVAT"
                        
                    results.append({
                        "Akcie": ticker,
                        "Aktuální Cena": f"${current_price:.2f}",
                        "Mám nakoupeno": f"{shares} ks",
                        "AI Pokyn pro zítřek": sig_text
                    })
                except Exception as e:
                    # Místo pouhého "Chyba dat" ukážeme konkrétní důvod (často API limit)
                    error_msg = str(e)[:40] + "..." if len(str(e)) > 40 else str(e)
                    results.append({"Akcie": ticker, "AI Pokyn pro zítřek": f"❌ {error_msg}"})
                    
                progress_bar.progress((i + 1) / len(assets))
                
            st.table(pd.DataFrame(results))
            st.warning("⚠️ **Pozor na limity dat:** Alpha Vantage zdarma dovoluje pouze 25 analýz denně. Každé kliknutí na tlačítko spotřebuje 1 analýzu za každou akcii. Pokud uvidíte chybu 'Limit 25 dotazů', musíte počkat do zítřka.")
        else:
            st.error("Chyba: Model nebyl nalezen.")
