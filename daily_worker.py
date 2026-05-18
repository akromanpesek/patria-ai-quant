import os
import pickle
import pandas as pd
from datetime import datetime
from data_pipeline import prepare_dataset

def get_daily_signal():
    print("=== START DAILY WORKER ===")
    
    # 1. Načtení vítězného modelu
    model_path = os.path.join('trained_models', 'best_deep_model.pkl')
    if not os.path.exists(model_path):
        return "Chyba: Model best_deep_model.pkl nenalezen!"
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model {model.name} úspěšně načten.")
    
    # 2. Stažení nejčerstvějších dat z burzy (např. posledních 60 dní pro indikátory)
    print("Stahuji dnešní živá tržní data...")
    try:
        # for_training=False zajistí, že nám systém neodmázne dnešní nejnovější svíčku!
        df = prepare_dataset(ticker="SPY", days_back=60, for_training=False)
    except Exception as e:
        return f"Chyba při stahování dat z burzy: {e}"
        
    # 3. Predikce pro dnešní den
    features = ['SMA_10', 'SMA_30', 'Volatility', 'RSI', 'Sentiment_Score']
    last_row = df.iloc[-1:]  # Vzít absolutně poslední aktuální svíčku
    today_date = last_row['Date'].iloc[0].strftime("%d. %m. %Y")
    current_price = last_row['Close'].iloc[0]
    
    # Výpočet predikce
    prediction = model.predict(last_row[features])[0]
    
    budget = 10000.00
    reserve_fee = 100.00
    allocation = budget - reserve_fee
    
    if prediction == 1:
        signal = "🟢 KOUPIT"
        action_text = (f"Cena SPY: ${current_price:.2f} (ISIN: US78462F1030)\n"
                       f"Model detekuje růstový trend.\n"
                       f"💰 ALOKACE: Nakupte za ${allocation:,.2f} (zůstatek ${reserve_fee} na poplatky).")
    else:
        signal = "🟡 DRŽET HOTOVOST / PRODAT"
        action_text = (f"Cena SPY: ${current_price:.2f} (ISIN: US78462F1030)\n"
                       f"Trh je nejistý nebo klesá. Zůstaň v hotovosti ($10,000.00).")
        
    # 4. Sestavení zprávy na mobil
    message = (
        f"🤖 Patria AI Quant - Denní Report\n"
        f"📅 Datum: {today_date}\n\n"
        f"🔥 SIGNÁL: {signal}\n"
        f"📊 {action_text}\n\n"
        f"🧠 (Generováno modelem: {model.name.split('_Gen')[0]} v cloudu)"
    )
    
    print("\n--- ZPRÁVA PRO ODESLÁNÍ ---")
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode('ascii', 'ignore').decode('ascii'))
    print("---------------------------\n")
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
