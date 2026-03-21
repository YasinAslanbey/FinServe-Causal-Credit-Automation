import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from dowhy import CausalModel 

data_file = r'C:/Users/yasla/Desktop/Task/Company_Database/full_credit_dataset.csv'
model_file = r'C:/Users/yasla/Desktop/Task/Company_Database/credit_model.pkl'

df = pd.read_csv(data_file)

df.columns = [c.replace('credit_history', 'credit_history_length') for c in df.columns]

X = df.drop('default', axis=1)
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

df['default_prob'] = rf.predict_proba(X)[:, 1]

def get_causal_effect_loan_term(new_user_data, past_df):
    
    confounders = ['income', 'credit_score', 'age', 'employment_years', 'debt_to_income']
    model = CausalModel(data=past_df, treatment='loan_term', outcome='default_prob', common_causes=confounders)
    
    id_est = model.identify_effect()
    est = model.estimate_effect(id_est, method_name="backdoor.linear_regression")
    
    unit_effect = est.value 
    current_term = new_user_data['loan_term'].values[0]
    best_term, min_risk_impact = current_term, 0

    for potential_term in [12, 24, 36, 48, 60]:
        effect = (potential_term - current_term) * unit_effect
        if effect < min_risk_impact:
            min_risk_impact, best_term = effect, potential_term
    return best_term, min_risk_impact

def get_causal_effect_loan_amount(new_user_data, past_df):
    
    confounders = ['income', 'credit_score', 'age', 'employment_years', 'debt_to_income']
    model = CausalModel(data=past_df, treatment='loan_amount', outcome='default_prob', common_causes=confounders)
    
    id_est = model.identify_effect()
    est = model.estimate_effect(id_est, method_name="backdoor.linear_regression")
    
    unit_effect = est.value 
    current_amt = new_user_data['loan_amount'].values[0]
    best_amt, min_risk_impact = current_amt, 0

    for pct in [0.9, 0.8, 0.7]: 
        potential_amt = current_amt * pct
        effect = (potential_amt - current_amt) * unit_effect
        if effect < min_risk_impact:
            min_risk_impact, best_amt = effect, potential_amt
    return best_amt, min_risk_impact

def get_causal_advice(new_user_data, past_df):
    
    b_term, t_impact = get_causal_effect_loan_term(new_user_data, past_df)
    b_amt, a_impact = get_causal_effect_loan_amount(new_user_data, past_df)
    
    c_term = new_user_data['loan_term'].values[0]
    c_amt = new_user_data['loan_amount'].values[0]

    advice_term = f"Strategy A: Reducing term to {b_term} months decreases risk by {abs(t_impact*100):.2f}%."
    advice_amt = f"Strategy B: Reducing amount to ${b_amt:,.0f} decreases risk by {abs(a_impact*100):.2f}%."
    
  
    return advice_term, advice_amt, t_impact, a_impact

with open(model_file, 'wb') as f:
    pickle.dump({'model': rf, 'enriched_df': df}, f)

print(f"Model Ready. Accuracy: {accuracy_score(y_test, rf.predict(X_test))*100:.2f}%")