import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score


data_file = r'C:/Users/yasla/Desktop/Task/Company_Database/full_credit_dataset.csv'
model_file = r'C:/Users/yasla/Desktop/Task/Company_Database/credit_model.pkl'


df = pd.read_csv(data_file)

df.columns = [c.replace('credit_history', 'credit_history_length') for c in df.columns]

X = df.drop('default', axis=1)
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

rf = grid_search.best_estimator_


y_prob = rf.predict_proba(X_test)[:, 1]
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")


def get_causal_advice(new_user_data, past_df):
    features = ['age', 'income', 'employment_years', 'debt', 'debt_to_income', 
                'credit_score', 'credit_history_length', 'loan_amount', 
                'loan_term', 'past_default', 'missed_payments']
    
    current_risk = rf.predict_proba(new_user_data[features])[0, 1]
    

    best_term, t_impact = new_user_data['loan_term'].values[0], 0
    for term in [12, 24, 48]:
        temp = new_user_data.copy()
        temp['loan_term'] = term
        impact = rf.predict_proba(temp[features])[0, 1] - current_risk
        if impact < t_impact: t_impact, best_term = impact, term
            
        
    best_amt, a_impact = new_user_data['loan_amount'].values[0], 0
    for pct in [0.8, 0.7]:
        temp = new_user_data.copy()
        temp['loan_amount'] *= pct
        impact = rf.predict_proba(temp[features])[0, 1] - current_risk
        if impact < a_impact: a_impact, best_amt = impact, temp['loan_amount'].values[0]

    advice_t = f"Strategy A: Reducing term to {best_term} months reduces risk."
    advice_a = f"Strategy B: Reducing amount to ${best_amt:,.0f} reduces risk."
    return advice_t, advice_a, t_impact, a_impact


# 6. SAVE
with open(model_file, 'wb') as f:
    pickle.dump({'model': rf, 'enriched_df': df}, f)
print("Model saved successfully.")
