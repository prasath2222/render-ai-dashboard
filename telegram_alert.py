
"""
Telegram alert sender.
Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as environment variables
or Streamlit secrets (secrets.toml).
"""

import os
import requests

def send_alert(message: str) -> bool:
    token   = os.getenv("8072713136:AAEE-xY6cZ-b6T5_HwZ2CESBqNB5AkNWtHE", "")
    chat_id = os.getenv("6941184785",   "")
    if not token or not chat_id:
        print("[Telegram] BOT_TOKEN or CHAT_ID not set — skipping alert.")
        return False
    try:
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": message,
                                        "parse_mode": "HTML"}, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"[Telegram] Error: {e}")
        return False
