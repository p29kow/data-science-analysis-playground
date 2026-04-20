import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

figure_prefix = "figures/hd-eda-"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

df = pd.read_csv(url, names=column_names, na_values='?')
print(f"Dataset shape:  {df.shape}")

df_clean = df.dropna(subset=["ca", "thal", "target"])
df_clean["target"] = (df_clean["target"] > 0).astype(int)

plt.figure(figsize=(6, 4))
sns.countplot(x="target", data=df_clean)
plt.title("Heart Disease Presence (No = 0, Yes = 1)")
plt.savefig(f"{figure_prefix}target-distribution.png")
plt.show()

plt.figure(figsize=(12, 10))
corr = df_clean.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.savefig(f"{figure_prefix}correlation-heatmap.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x="target", y="age", data=df_clean)
plt.title("Age Distribution by Heart Disease Presence")
plt.savefig(f"{figure_prefix}age-boxplot.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x="chol", y="thalach", hue="target", data=df_clean, alpha=0.7)
plt.title("Cholesterol vs Maximum Heart Rate by Heart Disease Presence")
plt.savefig(f"{figure_prefix}chol-thalach-scatter.png")
plt.show()

ct = pd.crosstab(df_clean["cp"], df_clean["target"], normalize="index")
ct.plot(kind="bar", stacked=True, figsize=(8, 5))
plt.title("Chest Pain Type Distribution by Heart Disease Presence")
plt.xlabel(
    "Chest Pain Type (Typical Angina=0, Atypical Angina=1, Non-anginal Pain=2, Asymptomatic=3)")
plt.ylabel("Proportion")
plt.savefig(f"{figure_prefix}cp-target-crosstab.png")
plt.show()

corr_target = df_clean.corr()["target"].sort_values(ascending=False)
print("Correlation with target variable:")
print(corr_target)