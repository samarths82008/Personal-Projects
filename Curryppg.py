import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_excel("stephen_curry_real_100_games.xlsx")  

# Optional: Clean/filter data if needed
df = df.dropna(subset=["Minutes Played", "Points"])

# Prepare input features and target
X = df[["Minutes Played"]]
y = df["Points"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Generate predictions
df["Predicted Points"] = model.predict(X)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Minutes Played", y="Points", label="Actual Points")
sns.lineplot(data=df, x="Minutes Played", y="Predicted Points", color="red", label="LSRL (Prediction)")
plt.title("Stephen Curry: Points vs. Minutes Played")
plt.xlabel("Minutes Played")
plt.ylabel("Points")
plt.legend()
plt.grid(True)
plt.show()

# Print model equation
slope = model.coef_[0]
intercept = model.intercept_
print(f"Regression equation: Points = {slope:.2f} * Minutes + {intercept:.2f}")
