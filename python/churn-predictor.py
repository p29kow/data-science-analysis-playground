import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib
import os
import urllib.request

figure_prefix = "figures/churn-pred-"
model_prefix = "models/churn-pred-"
data_path = 'datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv'
if not os.path.exists(data_path):
    url = 'https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/refs/heads/master/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    urllib.request.urlretrieve(url, data_path)
    print("Downloaded dataset.")

df = pd.read_csv(f'datasets/{data_path}')
print(f"Original shape: {df.shape}")

df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df.drop('Churn', axis=1)
y = df['Churn']

cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig(f'{figure_prefix}confusion_matrix.png')
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_proba):.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(f'{figure_prefix}roc_curve.png')
plt.show()

coeff_df = pd.DataFrame({'feature': X.columns, 'coefficient': model.coef_[0]})
coeff_df = coeff_df.sort_values('coefficient', ascending=False)
plt.figure(figsize=(10,8))
sns.barplot(x='coefficient', y='feature', data=coeff_df.head(10))
plt.title('Top 10 Features Driving Churn')
plt.tight_layout()
plt.savefig(f'{figure_prefix}feature_importance.png')
plt.show()

print("\nTop 3 churn drivers:")
print(coeff_df.head(3))

joblib.dump(model, f'{model_prefix}logistic_model.pkl')
joblib.dump(scaler, f'{model_prefix}scaler.pkl')
print(f"\nModel and scaler saved as '{model_prefix}logistic_model.pkl' and '{model_prefix}scaler.pkl'")