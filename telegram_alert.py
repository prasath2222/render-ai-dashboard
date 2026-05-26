# =========================================================
# TELEGRAM ALERT SYSTEM
# =========================================================
# FILE:
#
# telegram_alert.py
#
# RUN:
#
# python telegram_alert.py
# =========================================================

import requests

# =========================================================
# REPLACE THESE
# =========================================================

BOT_TOKEN = "8819680977:AAG3ed3_51lx5vJ4d9Ou0DBq1S8mvd2-rag"

CHAT_ID = "6941184785"

# =========================================================
# SEND ALERT FUNCTION
# =========================================================

def send_alert(

    coin,
    signal,
    confidence,
    price

):

    message = f"""
🚀 AI CRYPTO ALERT

Coin: {coin}

Signal: {signal}

Confidence: {confidence:.2f}%

Price: ${price:.2f}
"""

    url = (
        f"https://api.telegram.org/bot"
        f"{BOT_TOKEN}/sendMessage"
    )

    payload = {

        "chat_id": CHAT_ID,

        "text": message
    }

    response = requests.post(
        url,
        data=payload
    )

    print(response.text)

# =========================================================
# TEST ALERT
# =========================================================

send_alert(

    coin="BTC",

    signal="BUY",

    confidence=78.4,

    price=68500
)
