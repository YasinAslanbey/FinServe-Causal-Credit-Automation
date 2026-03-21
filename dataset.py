import pandas as pd
import numpy as np
import os

path = r'C:\CS50'
if not os.path.exists(path): os.makedirs(path)

np.random.seed(42)
n = 1000

# 1 time info
age = np.random.randint(22, 65, n)
employment_years = (np.random.uniform(0.1, 0.8, n) * (age - 18)).astype(int)
credit_history_length = (np.random.uniform(0.2, 0.9, n) * (age - 18)).astype(int)

# 2 financials
income = np.random.randint(25000, 180000, n)
debt = (income * np.random.uniform(0.05, 0.6, n)).astype(int) # adding some noise
debt_to_income = np.round(debt / income, 2)

# 3 loan info
loan_amount = (income * np.random.uniform(0.5, 3.0, n)).astype(int)
loan_term = np.random.choice([12, 24, 36, 48, 60], n)

# 4. behavior , risk score
past_default = np.random.choice([0, 1], n, p=[0.9, 0.1])
missed_payments = np.random.poisson(0.5, n) 
missed_payments[past_default == 1] += np.random.randint(1, 4, sum(past_default))

# credit score corelation
base_score = 650 + (employment_years * 2) + (credit_history_length * 2)
penalty = (missed_payments * 40) + (past_default * 100) + (debt_to_income * 150)
credit_score = np.clip(base_score - penalty + np.random.normal(0, 30, n), 300, 850).astype(int)

# target = default 
risk_score = (debt_to_income * 6) + (past_default * 3) + (missed_payments * 0.8) - (credit_score / 150)
prob = 1 / (1 + np.exp(-(risk_score - 2)))
default = (prob > 0.5).astype(int)


df = pd.DataFrame({
    'age': age, 'income': income, 'employment_years': employment_years,
    'debt': debt, 'debt_to_income': debt_to_income, 'credit_score': credit_score,
    'credit_history_length': credit_history_length, 'loan_amount': loan_amount,
    'loan_term': loan_term, 'past_default': past_default, 
    'missed_payments': missed_payments, 'default': default
})

df.to_csv(os.path.join(path, 'full_credit_dataset.csv'), index=False)
print(f"Dataset hazır: {path}\\full_credit_dataset.csv")