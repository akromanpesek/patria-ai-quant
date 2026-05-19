import os
import pickle
import pandas as pd
from datetime import datetime
from data_pipeline import prepare_dataset

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
        f"🤖 <b>Patria AI Quant - Multi-Asset Report</b>",
        f"📅 Datum: {today_date}",
        f"💰 Hotovost k dispozici: ${cash:,.2f}\n",
        f"<b>ANALÝZA VAŠEHO PORTFOLIA:</b>"
    ]
    
    features = ['SMA_10', 'SMA_30', 'Volatility', 'RSI', 'Sentiment_Score']
    
    for asset in assets:
        ticker = asset["ticker"]
        shares = asset.get("shares", 0)
        buy_price = asset.get("buy_price", 0)
        
        print(f"Zpracovávám: {ticker}...")
        try:
            df = prepare_dataset(ticker=ticker, days_back=60, for_training=False)
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
