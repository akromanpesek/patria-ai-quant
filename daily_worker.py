import os
import pickle
import pandas as pd
from datetime import datetime
import importlib
import data_pipeline
importlib.reload(data_pipeline)

def get_daily_signal():
    print("=== START MULTI-ASSET WORKER ===")
    
    # 1. Načtení modelu
    model_path = os.path.join('trained_models', 'best_deep_model.pkl')
    if not os.path.exists(model_path):
        return "Chyba: Model best_deep_model.pkl nenalezen!"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # 2. Načtení portfolia
    import json
    portfolio_path = "portfolio.json"
    if not os.path.exists(portfolio_path):
        return "Chyba: Soubor portfolio.json nenalezen!"
        
    with open(portfolio_path, "r", encoding="utf-8") as f:
        portfolio = json.load(f)
        
    assets = portfolio.get("assets", [])
    cash = portfolio.get("cash", 0)
    
    today_date = datetime.now().strftime("%d. %m. %Y")
    
    message_lines = [
        f"🤖 <b>Patria AI Quant - Kompletní Report</b>",
        f"📅 Datum: {today_date}",
        f"💰 Hotovost k dispozici: ${cash:,.2f}\n"
    ]
    
    features = ['SMA_10', 'SMA_30', 'Volatility', 'RSI', 'Sentiment_Score']
    
    # --- HLAVNÍ STRATEGIE: SPY ---
    try:
        spy_df = data_pipeline.prepare_dataset(ticker="SPY", days_back=60, for_training=False)
        spy_last = spy_df.iloc[-1:]
        spy_price = spy_last['Close'].iloc[0]
        spy_pred = model.predict(spy_last[features])[0]
        
        if spy_pred == 1:
            spy_sig = "🟢 KOUPIT SPY"
        else:
            spy_sig = "🔴 DRŽET HOTOVOST / PRODAT SPY"
            
        message_lines.append(f"🔥 <b>HLAVNÍ TRH (S&P 500)</b>")
        message_lines.append(f"Cena SPY: ${spy_price:.2f}")
        message_lines.append(f"Pokyn: <b>{spy_sig}</b>\n")
    except Exception as e:
        message_lines.append(f"🔥 <b>HLAVNÍ TRH (S&P 500)</b>: ❌ Chyba dat\n")

    message_lines.append(f"<b>ANALÝZA VAŠEHO PORTFOLIA:</b>")
    
    # --- VEDLEJŠÍ STRATEGIE: OSOBNÍ AKCIE ---
    for asset in assets:
        ticker = asset["ticker"]
        shares = asset.get("shares", 0)
        buy_price = asset.get("buy_price", 0)
        
        print(f"Zpracovávám: {ticker}...")
        try:
            df = data_pipeline.prepare_dataset(ticker=ticker, days_back=60, for_training=False)
            last_row = df.iloc[-1:]
            current_price = last_row['Close'].iloc[0]
            prediction = model.predict(last_row[features])[0]
            
            # Výpočet profitu
            if shares > 0 and buy_price > 0:
                profit = (current_price - buy_price) * shares
                profit_str = f"+${profit:.2f}" if profit >= 0 else f"-${abs(profit):.2f}"
                pos_info = f"Máte: {shares}ks (Zisk: {profit_str})"
            else:
                pos_info = "Nemáte nakoupeno"
            
            # Signál
            if prediction == 1:
                if shares > 0:
                    sig_text = "🟢 DRŽET (Pozice roste)"
                else:
                    sig_text = "🟢 KOUPIT (Nový trend)"
            else:
                if shares > 0:
                    sig_text = "🔴 PRODAT (Ukončit pozici)"
                else:
                    sig_text = "🟡 IGNOROVAT (Klesá)"
                    
            line = f"▪️ <b>{ticker}</b> (${current_price:.2f}): {sig_text} | <i>{pos_info}</i>"
            message_lines.append(line)
            
        except Exception as e:
            message_lines.append(f"▪️ <b>{ticker}</b>: ❌ Chyba dat ({str(e)[:30]})")
            
    message_lines.append(f"\n🧠 Model: {model.name.split('_Gen')[0]}")
    
    message = "\n".join(message_lines)
    print(message)
    return message

def send_telegram_message(message):
    import requests
    from dotenv import load_dotenv
    load_dotenv()
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        print("Chyba: Chybí TELEGRAM_BOT_TOKEN nebo TELEGRAM_CHAT_ID v souboru .env")
        return
        
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("✅ Telegram zpráva byla úspěšně odeslána na váš mobil!")
        else:
            print(f"❌ Chyba při odesílání: {response.text}")
    except Exception as e:
        print(f"❌ Výjimka při odesílání: {e}")

if __name__ == "__main__":
    msg = get_daily_signal()
    # Odeslání zprávy do mobilu uživatele
    send_telegram_message(msg)
