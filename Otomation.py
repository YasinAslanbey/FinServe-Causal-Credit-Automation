import os
import json
import time
import pandas as pd
import pickle
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


WATCH_DIRECTORY = r'C:/Users/yasla/Desktop/Task/Json_input'
MODEL_FILE = r'C:/Users/yasla/Desktop/Task/Company_Database/credit_model.pkl'

PROCESSED_DATA_FILE = r'C:/Users/yasla/Desktop/Task/Company_Database/processed_applications.csv'
MODULE_PATH = r'C:/Users/yasla/Desktop/Task'

sys.path.append(MODULE_PATH)

try:
    from ML_CASUAL import get_causal_advice
except ImportError:
    print("Error: ML_CASUAL.py not found.")

class ApplicationHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.json'):
            return
        
        print(f"\n[SYSTEM]: Processing: {os.path.basename(event.src_path)}")
        time.sleep(1)
        
        try:

            with open(MODEL_FILE, 'rb') as f:
                saved = pickle.load(f)
            
            rf_model = saved['model']
            past_df = saved['enriched_df']
            
            with open(event.src_path, 'r') as f:
                new_app_dict = json.load(f)
            

            if "credit_history" in new_app_dict:
                new_app_dict["credit_history_length_length"] = new_app_dict.pop("credit_history")
            elif "credit_history_length" in new_app_dict:
                 new_app_dict["credit_history_length_length"] = new_app_dict.pop("credit_history_length")

            new_app_df = pd.DataFrame([new_app_dict])

         
            features = ['age', 'income', 'employment_years', 'debt', 'debt_to_income', 
                        'credit_score', 'credit_history_length_length', 'loan_amount', 
                        'loan_term', 'past_default', 'missed_payments']
            
            
            new_app_df_filtered = new_app_df[features]
            
            # 5. Tahmin
            risk_prob = rf_model.predict_proba(new_app_df_filtered)[0, 1]
            status = "Approved" if risk_prob < 0.20 else "Manual Review"
            
            print(f"Risk: {risk_prob*100:.2f}% | Decision: {status}")

 
            advice_t, advice_a, imp_t, imp_a = "N/A", "N/A", 0.0, 0.0

            if status == "Manual Review":
                advice_t, advice_a, imp_t, imp_a = get_causal_advice(new_app_df, past_df)
                
                print("\n" + "="*45)
                print("🔍 CAUSAL IMPACT ANALYSIS")
                print("="*45)
                print(f"Term Reduction Impact  : %{abs(imp_t*100):.2f} lower risk")
                print(f"Amount Reduction Impact: %{abs(imp_a*100):.2f} lower risk")
                print(f"\n💡 {advice_t}") 
                print(f"💡 {advice_a}")
                print("="*45 + "\n")



            new_app_df['default_prob'] = risk_prob
            new_app_df['causal_strategy_term'] = advice_t
            new_app_df['causal_strategy_amount'] = advice_a
            new_app_df['processed_at'] = time.ctime()
            new_app_df['application_status'] = status

            file_exists = os.path.isfile(PROCESSED_DATA_FILE)
            new_app_df.to_csv(PROCESSED_DATA_FILE, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
            
            print(f"[SUCCESS]: Data saved to processed_applications.csv")

        except Exception as e:
            print(f"[ERROR]: {e}")

if __name__ == "__main__":
    event_handler = ApplicationHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=False)
    print(f"System Online. Monitoring: {WATCH_DIRECTORY}")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()