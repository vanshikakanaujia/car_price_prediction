import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
N = 300
#give data
brands   = ["Maruti", "Hyundai", "Honda", "Toyota", "Ford"]
fuels    = ["Petrol", "Diesel", "CNG"]
transmit = ["Manual", "Automatic"]
brand=np.random.choice(brands, N)
year= np.random.randint(2010, 2024, N)
km_driven=np.random.randint(5000, 150000, N)
fuel=np.random.choice(fuels, N, p=[0.55, 0.35, 0.10])
transmission=np.random.choice(transmit, N, p=[0.65, 0.35])
engine_cc=np.random.choice([800, 1000, 1200, 1500, 1800, 2000], N)
seats= np.random.choice([5, 7], N, p=[0.75, 0.25])
age=2024 - year
base_price=(engine_cc / 200) + (4 if transmission == "Automatic" else 0).mean() if False else \
             np.array([(cc / 200) + (4 if t == "Automatic" else 0) for cc, t in zip(engine_cc, transmission)])
price=(
base_price
- age * 0.6
- km_driven / 30000
+ np.where(fuel == "Diesel", 1.5, 0)
+ np.where(np.isin(brand, ["Toyota", "Honda"]), 1.0, 0)
+ np.random.normal(0, 0.8, N)
).clip(1.0, 25.0).round(2)
df = pd.DataFrame({
    "Brand":brand,
    "Year":year,
    "KM_Driven":km_driven,
    "Fuel":fuel,
    "Transmission":transmission,
    "Engine_CC":engine_cc,
    "Seats":seats,
    "Price_Lakh":price
})
print(f"\n[DATA] Dataset: {len(df)} cars")
print(f"Price range: ₹{price.min():.1f}L — ₹{price.max():.1f}L")
print(f"Average price: ₹{price.mean():.2f}L")
print(f"\n[DATA] Sample:\n{df.head(4).to_string(index=False)}")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(df["Price_Lakh"], bins=20, color="#3498DB", edgecolor="black")
axes[0].set_title("Car Price Distribution")
axes[0].set_xlabel("Price (₹ Lakhs)")
axes[0].set_ylabel("Count")
fuel_avg = df.groupby("Fuel")["Price_Lakh"].mean().sort_values()
axes[1].barh(fuel_avg.index, fuel_avg.values, color=["#E74C3C","#2ECC71","#F39C12"], edgecolor="black")
axes[1].set_title("Average Price by Fuel Type")
axes[1].set_xlabel("Average Price (₹ Lakhs)")
axes[2].scatter(df["KM_Driven"], df["Price_Lakh"], alpha=0.4, color="#9B59B6", edgecolors="none")
axes[2].set_title("KM Driven vs Price")
axes[2].set_xlabel("KM Driven")
axes[2].set_ylabel("Price (₹ Lakhs)")
plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
plt.close()
print("\n[EDA] Plots saved → eda_plots.png")
df_model = df.copy()
le = LabelEncoder()
for col in ["Brand", "Fuel", "Transmission"]:
    df_model[col] = le.fit_transform(df_model[col])
X = df_model.drop("Price_Lakh", axis=1)
y = df_model["Price_Lakh"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")
models = {
"Linear Regression":LinearRegression(),
"Decision Tree":DecisionTreeRegressor(max_depth=5, random_state=42),
"Random Forest":RandomForestRegressor(n_estimators=100, random_state=42),
}
results = {}
print("\n[TRAINING] Results:\n")
print(f"  {'Model':<22} {'MAE':>8} {'RMSE':>8} {'R² Score':>10}")
for name, model in models.items():
model.fit(X_train, y_train)
preds=odel.predict(X_test)
mae=mean_absolute_error(y_test, preds)
rmse=np.sqrt(mean_squared_error(y_test, preds))
r2=r2_score(y_test, preds)
results[name] = {"model": model, "preds": preds, "mae": mae, "rmse": rmse, "r2": r2}
print(f"  {name:<22} {mae:>7.2f}L {rmse:>7.2f}L {r2:>10.3f}")
best_name = max(results, key=lambda k: results[k]["r2"])
best = results[best_name]
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_test, best["preds"], alpha=0.6, color="#E74C3C", edgecolors="none")
lims = [min(y_test.min(), best["preds"].min()), max(y_test.max(), best["preds"].max())]
ax.plot(lims, lims, "k--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Price (₹ Lakhs)")
ax.set_ylabel("Predicted Price (₹ Lakhs)")
ax.set_title(f"Actual vs Predicted — {best_name}")
ax.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()
print(f"\n[RESULT] Best model: {best_name}  (R² = {best['r2']:.3f})")
print("[RESULT] Actual vs Predicted chart saved → actual_vs_predicted.png")
rf_model=results["Random Forest"]["model"]
importances=pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
fig, ax=plt.subplots(figsize=(7, 4))
importances.plot(kind="barh", ax=ax, color="#3498DB", edgecolor="black")
ax.set_title("Feature Importance (Random Forest)")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("[RESULT] Feature importance chart saved → feature_importance.png")
print("PREDICT PRICE FOR A NEW CAR")
new_car = pd.DataFrame([{
"Brand":1,
"Year":2019,
"KM_Driven":45000,
"Fuel":2,
"Transmission":0,
"Engine_CC":1200,
"Seats":5,
}])
predicted_price = results["Random Forest"]["model"].predict(new_car)[0]
print(f"\n  Honda | 2019 | 45,000 KM | Petrol | Manual | 1200cc | 5 seats")
print(f"\n  Predicted Price : ₹ {predicted_price:.2f} Lakhs")