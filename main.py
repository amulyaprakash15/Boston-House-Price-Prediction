# main.py

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load dataset
try:
    df = pd.read_csv('boston.csv')
except FileNotFoundError:
    print("❌ ERROR: Dataset 'boston.csv' not found in your folder. Please check the name or path.")
    exit()

# 3. Quick check
print("✅ First 5 rows:\n", df.head())
print("✅ Shape:", df.shape)
print("✅ Missing values:\n", df.isnull().sum())

# 4. Drop missing values (if any)
df = df.dropna()

# 5. Visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 6. Split features and target
if 'MEDV' not in df.columns:
    print("❌ ERROR: 'MEDV' column (house price) not found. Please check your dataset's target column.")
    exit()

X = df.drop(columns='MEDV')
y = df['MEDV']

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Initialize & train model
model = LinearRegression()
model.fit(X_train, y_train)

# 9. Predict on test set
y_pred = model.predict(X_test)

# 10. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📈 Model Evaluation:")
print(f"🔹 Mean Squared Error (MSE): {mse:.2f}")
print(f"🔹 Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"🔹 R-squared (R² Score): {r2:.2f}")

# 11. Predict on a sample row
sample_input = X_test.iloc[0].values.reshape(1, -1)
sample_prediction = model.predict(sample_input)
print(f"\n🏠 Predicted price for one test sample: ${sample_prediction[0]:.2f}")
