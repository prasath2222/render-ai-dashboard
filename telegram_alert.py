import requests

# =========================================================
# TELEGRAM CONFIG
# =========================================================

BOT_TOKEN = "8072713136:AAEE-xY6cZ-b6T5_HwZ2CESBqNB5AkNWtHE"

CHAT_ID = "6941184785"

# =========================================================
# SEND ALERT
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
