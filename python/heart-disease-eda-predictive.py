import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

figure_prefix = "figures/hd-eda-"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

df = pd.read_csv(url, names=column_names, na_values='?')
print(f"Dataset shape:  {df.shape}")

df_clean = df.dropna(subset=["ca", "thal", "target"])
df_clean["target"] = (df_clean["target"] > 0).astype(int)

X = df_clean.drop("target", axis=1)
y = df_clean["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Logistic Regression Accuracy: {accuracy:.2f}")