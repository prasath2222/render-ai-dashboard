pip install schedule
python auto_run.py
import schedule
import time
import os

# =========================================================
# RUN AI
# =========================================================

def run_ai():

    print("\nRUNNING AI...\n")

    os.system(
        "python predict.py"
    )

# =========================================================
# SCHEDULE
# =========================================================

schedule.every(1).hours.do(
    run_ai
)

# =========================================================
# START
# =========================================================

print("AI AUTO SYSTEM STARTED")

# RUN FIRST TIME
run_ai()

# LOOP
while True:

    schedule.run_pending()

    time.sleep(1)
