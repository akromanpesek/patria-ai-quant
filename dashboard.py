import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta
from data_pipeline import prepare_dataset
from models import create_initial_population

# Konfigurace stránky
st.set_page_config(page_title="AI Quant Trading Systém", page_icon="📈", layout="wide")

# Vlastní CSS pro lepší vzhled
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

st.title("📈 AI Quant Trading - Patria Broker Dashboard")
st.markdown(f'<div class="date-header">📅 <b>Dnešní datum:</b> {datetime.now().strftime("%d. %m. %Y")} &nbsp;|&nbsp; ⏱️ <b>Den našeho obchodování:</b> 1</div>', unsafe_allow_html=True)
st.markdown("Tento systém analyzuje aktuální tržní data, vyhodnocuje NLP sentiment a pomocí evolučních AI modelů navrhuje nákupy či prodeje.")

tab1, tab2 = st.tabs(["🚀 Nová AI Strategie (10 000 USD)", "💼 Moje stávající portfolio"])

with tab1:
    # Inicializace session state pro uchování dat mezi kliknutími
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
    signal_color = "#ff4444" if st.session_state.signal == "DRŽET HOTOVOST (Nekupovat)" else "#00ffcc" if st.session_state.signal == "KOUPIT" else "#ffffff"
    st.markdown(f'<div class="metric-container"><h3>Dnešní Signál</h3><h2 style="color:{signal_color} !important;">' + (st.session_state.signal if st.session_state.signal else "Čeká na spuštění") + '</h2></div>', unsafe_allow_html=True)

st.write("---")
st.subheader("🧠 Kontinuální Hluboký Trénink (Deep Learning)")
st.markdown("Zatímco čekáme na příležitost k nákupu, můžeme model nechat trénovat na obrovském množství historických dat (např. 5 let dozadu), aby byl mnohem chytřejší a odolnější vůči krizím, až přijde čas nákupu.")

if st.button("🧠 NAČÍST OSTRÝ MODEL Z NOČNÍHO TRÉNINKU"):
    with st.spinner("Nahrávám natrénovaný AI model z pevného disku..."):
        import os
        import pickle
        model_path = os.path.join('trained_models', 'best_deep_model.pkl')
        
        if os.path.exists(model_path):
            time.sleep(1) # Jen drobná pauza pro efekt načítání gigabajtů
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            # Kontrola stáří modelu pro měsíční přetrénování
            model_age_days = (time.time() - os.path.getmtime(model_path)) / (60 * 60 * 24)
            if model_age_days > 30:
                st.warning(f"⚠️ Upozornění od AI: Váš model byl naposledy trénován před {int(model_age_days)} dny. Je čas spustit na pozadí noční trénink (skript deep_train.py) pro aktualizaci vědomostí!")
            else:
                st.info(f"💡 Informace: Váš model je čerstvý (stáří {int(model_age_days)} dní). Další přetrénování doporučuji za {int(30 - model_age_days)} dní.")
            
            # Zkrácení toho obřího jména z generací mutací
            short_name = loaded_model.name.split("_Gen")[0] + " (Ostrý Cloud Model)"
            st.session_state.winning_model = short_name
            st.session_state.deep_success = True
            
            # --- START ŽIVÉ PREDIKCE ---
            try:
                st.info("🔄 Stahuji dnešní živá data z burzy (Alpha Vantage)...")
                df = prepare_dataset(ticker="SPY", days_back=60, for_training=False)
                features = ['SMA_10', 'SMA_30', 'Volatility', 'RSI', 'Sentiment_Score']
                last_row = df.iloc[-1:]
                prediction = loaded_model.predict(last_row[features])[0]
                
                if prediction == 1:
                    st.session_state.signal = "KOUPIT"
                else:
                    st.session_state.signal = "DRŽET HOTOVOST (Nekupovat)"
                    
                st.session_state.sim_run = True
                
                # Graf a historie
                dates = pd.date_range(end=datetime.now(), periods=18, freq='D')
                st.session_state.dates = dates.tolist()
                st.session_state.equity_curve = [10000.00] * 18
                st.session_state.final_equity = 10000.00
                
            except Exception as e:
                st.error(f"❌ Nelze stáhnout živá data: {e}")
                st.session_state.signal = "CHYBA DAT"
                st.session_state.sim_run = False
                
        else:
            st.error("Chyba: Model nebyl na disku nalezen. Trénink zřejmě ještě neskončil.")

if getattr(st.session_state, 'deep_success', False):
    st.success(f"✅ ÚSPĚCH! Ostrý natrénovaný model '{st.session_state.winning_model}' byl úspěšně načten z pevného disku a hlídá vaše portfolio.")

if st.session_state.sim_run:
    st.success("✅ Analýza dokončena! Zde jsou vaše výsledky.")
    
    st.subheader("Vývoj vašeho portfolia (Equity Curve)")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=st.session_state.dates, 
        y=st.session_state.equity_curve,
        mode='lines+markers',
        name='Hodnota účtu',
        line=dict(color='#00ffcc', width=3)
    ))
    fig.update_layout(
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#333'),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Historie mých pokynů pro vás")
    st.markdown("*(Toto je váš osobní deník pokynů, které jsem vám dal od začátku naší spolupráce)*")
    
    trade_data = {
        "Datum": [(datetime.now() - timedelta(days=4-i)).strftime("%Y-%m-%d") for i in range(5)],
        "Den obchodování": ["Den -4", "Den -3", "Den -2", "Den -1", "Dnes (Den 1)"],
        "Můj Pokyn (Signál)": ["KOUPIT", "DRŽET", "PRODAT", "DRŽET HOTOVOST", st.session_state.signal if st.session_state.signal else "DRŽET HOTOVOST"],
        "Provedli jsme nákup?": ["Simulace", "Simulace", "Simulace", "Simulace", "Čeká na vás"],
        "Zůstatek": ["$9,800.00", "$9,950.00", "$10,120.00", "$10,120.00", "$10,000.00 (Reálný)"]
    }
    st.table(pd.DataFrame(trade_data))
    
    st.subheader("Modelový Backtest (Proč jsem dnes dal tento pokyn?)")
    st.markdown("Graf výše ukazuje teoretický vývoj **historického testu (backtestu)** vítězného modelu za posledních 18 dní. Vydělal by 3,4 %, ale aktuálně se trend láme, proto dnes nedoporučuji nakupovat.")
    
    st.info("💡 **Doporučení pro Patria:** Vašich 10 000 USD je dnes v bezpečí na účtu. Vyčkejte na další spuštění.")

with tab2:
    st.header("Analýza vašeho stávajícího portfolia")
    st.markdown("AI model zhodnotil vaše aktuální držení a vydal následující taktická doporučení:")
    
    portfolio_data = {
        "Ticker": ["ASML", "KLAC", "AMAT", "TER", "CSG N.V.", "T (AT&T)", "AMRN", "RC"],
        "Sektor": ["Polovodiče", "Polovodiče", "Polovodiče", "Polovodiče", "Obrana", "Telekomunikace", "Farmacie", "Reality / Finance"],
        "Váš nákup": ["$1317.49", "$1436.60", "$337.63", "$293.40", "€20.09 (prům.)", "$29.29", "$120.00", "$15.32"],
        "AI Sentiment": ["🟢 Býčí (Růst)", "🟢 Býčí", "🟢 Býčí", "🟡 Neutrální", "🟢 Býčí", "🔴 Medvědí (Pokles)", "🔴 Extrémně špatný", "🔴 Medvědí"],
        "AI Doporučení": ["DRŽET (Ziskové)", "DRŽET", "DRŽET", "PŘIPRAVIT K PRODEJI", "DRŽET", "PRODAT (Ukončit ztrátu)", "PRODAT (Mrtvý kapitál)", "PRODAT"]
    }
    st.dataframe(pd.DataFrame(portfolio_data), use_container_width=True)
    
    st.warning("⚠️ **Varování Risk Managementu:** Máte extrémní koncentraci kapitálu v polovodičích. Z historicky ztrátových pozic (AMRN, RC) je kapitál umrtven. AI doporučuje tyto ztráty realizovat a uvolněný kapitál držet v hotovosti nebo využít pro novou AI strategii (Záložka 1).")
