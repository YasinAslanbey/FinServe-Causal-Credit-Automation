import os, json, time, pandas as pd, pickle, sys, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


WATCH_DIR = r'C:/Users/yasla/Desktop/Task/Json_input'
MODEL_PATH = r'C:/Users/yasla/Desktop/Task/Company_Database/credit_model.pkl'
LOG_FILE = r'C:/Users/yasla/Desktop/Task/Company_Database/processed_applications.csv'
sys.path.append(r'C:/Users/yasla/Desktop/Task')

from ML_CASUAL import get_causal_advice

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.src_path.endswith('.json'): return
        print(f"\n[SYSTEM]: Processing {os.path.basename(event.src_path)}...")
        time.sleep(1)
        
        try:
            with open(MODEL_PATH, 'rb') as f:
                saved = pickle.load(f)
            model, past_df = saved['model'], saved['enriched_df']
            
            with open(event.src_path, 'r') as f:
                data = json.load(f)
            
        
            if "credit_history" in data:
                data["credit_history_length"] = data.pop("credit_history")
            
            df_new = pd.DataFrame([data])
            cols = ['age', 'income', 'employment_years', 'debt', 'debt_to_income', 
                    'credit_score', 'credit_history_length', 'loan_amount', 
                    'loan_term', 'past_default', 'missed_payments']
            
           
            risk = model.predict_proba(df_new[cols])[0, 1]
            status = "Approved" if risk < 0.20 else "Manual Review"
            
            adv_t, adv_a = "N/A", "N/A"
            if status == "Manual Review":
                adv_t, adv_a, _, _ = get_causal_advice(df_new[cols], past_df)

            
            df_new['risk_score'], df_new['status'] = risk, status
            df_new['advice_term'], df_new['advice_amount'] = adv_t, adv_a
            df_new['time'] = time.ctime()
            
            order = cols + ['risk_score', 'status', 'advice_term', 'advice_amount', 'time']
            df_new[order].to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False, encoding='utf-8-sig')
            
            print(f"[SUCCESS]: Risk %{risk*100:.2f} | Status: {status}")

        except Exception as e: print(f"[ERROR]: {e}")

if __name__ == "__main__":
    observer = Observer()
    observer.schedule(Handler(), WATCH_DIR, recursive=False)
    observer.start()
    print(f"Monitoring: {WATCH_DIR}")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: observer.stop()
    observer.join()
