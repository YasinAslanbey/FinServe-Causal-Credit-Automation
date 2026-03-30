import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score

data_file = r'C:/Users/yasla/Desktop/Task/Company_Database/full_credit_dataset.csv'
model_file = r'C:/Users/yasla/Desktop/Task/Company_Database/credit_model.pkl'

df = pd.read_csv(data_file)

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


print("\n" + "="*30)
print("--- Feature Importance (Top Drivers) ---")
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importance.head(10))
print("="*30 + "\n")


y_prob = rf.predict_proba(X_test)[:, 1]
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

def get_causal_advice(new_user_data, past_df):
    features = ['age', 'income', 'employment_years', 'debt', 'debt_to_income', 
                'credit_score', 'credit_history_length', 'loan_amount', 
                'loan_term', 'past_default', 'missed_payments']
    
    current_risk = rf.predict_proba(new_user_data[features])[0, 1]
    
    best_t_impact = 0
    best_term = new_user_data['loan_term'].values[0]
    
    for t in [12, 24, 36]: 
        temp_t = new_user_data.copy()
        temp_t['loan_term'] = t
        risk_t = rf.predict_proba(temp_t[features])[0, 1]
        impact = (risk_t - current_risk) * 100
        if impact < best_t_impact: 
            best_t_impact = impact
            best_term = t
            
    best_a_impact = 0
    best_amt = new_user_data['loan_amount'].values[0]
    
    for ratio in [0.9, 0.8, 0.7]: 
        temp_a = new_user_data.copy()
        temp_a['loan_amount'] *= ratio
        risk_a = rf.predict_proba(temp_a[features])[0, 1]
        impact = (risk_a - current_risk) * 100
        if impact < best_a_impact:
            best_a_impact = impact
            best_amt = temp_a['loan_amount'].values[0]

    advice_t = f"Reduce term to {best_term}m (Risk: {best_t_impact:.2f}%)"
    advice_a = f"Reduce amount to ${best_amt:,.0f} (Risk: {best_a_impact:.2f}%)"
    
    return advice_t, advice_a, best_t_impact, best_a_impact

with open(model_file, 'wb') as f:
    pickle.dump({'model': rf, 'enriched_df': df}, f)
print("Model saved successfully.")
