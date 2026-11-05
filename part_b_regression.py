"""Part B - delivery-hours regression."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["delivery_hours"]

# 1. split 840 / 180 / 180
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# 2. scaling without leakage: fit on training rows only
scaler = StandardScaler().fit(X_train)
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)
train_mean = float(np.abs(train_scaled.mean(axis=0)).max())
test_mean = float(np.abs(test_scaled.mean(axis=0)).max())
pd.DataFrame({"metric": ["train_rows", "val_rows", "test_rows",
                         "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
              "value": [str(len(X_train)), str(len(X_val)), str(len(X_test)),
                        f"{round(train_mean, 4):g}", f"{round(test_mean, 4):g}"]}
             ).to_csv("outputs/part_b_checks.csv", index=False)


def scores(model_name, split, y_true, pred):
    mse = mean_squared_error(y_true, pred)
    return {"model": model_name, "split": split, "MAE": mean_absolute_error(y_true, pred),
            "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, pred)}


rows = []
# 3. baseline
rows.append(scores("mean_baseline", "val", y_val, np.full(len(y_val), y_train.mean())))

# 4. tree-depth diagnostic
depths = [1, 2, 3, 4, 6, 8, 12, None]
depth_rows = []
for d in depths:
    t = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    depth_rows.append({"max_depth": "unlimited" if d is None else d,
                       "train_rmse": float(np.sqrt(mean_squared_error(y_train, t.predict(X_train)))),
                       "val_rmse": float(np.sqrt(mean_squared_error(y_val, t.predict(X_val)))),
                       "_d": d})
depth_df = pd.DataFrame(depth_rows)
# lowest val_rmse; ties -> smaller depth (unlimited counts as largest)
order = sorted(range(len(depths)), key=lambda i: (round(depth_df.loc[i, "val_rmse"], 10),
                                                  10**9 if depths[i] is None else depths[i]))
best_depth = depths[order[0]]
depth_df.drop(columns="_d").round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)

# 5. four candidates
candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
val_rmse = {}
for name, m in candidates.items():
    m.fit(X_train, y_train)
    r = scores(name, "val", y_val, m.predict(X_val))
    rows.append(r)
    val_rmse[name] = r["RMSE"]

# 6. learning-rate experiment
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    p = g.predict(X_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": float(np.sqrt(mean_squared_error(y_val, p))),
                    "val_r2": r2_score(y_val, p)})
pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# 7. cross-validated comparison of the two best + final model
ranking = sorted(val_rmse, key=val_rmse.get)
top2 = ranking[:2]
kf = KFold(n_splits=5, shuffle=True, random_state=42)       # one shared object
X_trval = pd.concat([X_train, X_val])
y_trval = pd.concat([y_train, y_val])
cv_rows = []
for name in top2:
    s = cross_val_score(candidates[name], X_trval, y_trval, cv=kf)
    cv_rows.append({"model": name, "mean_r2": s.mean(), "std_r2": s.std()})
pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)

final_name = ranking[0]
rows.append(scores(final_name, "test", y_test, candidates[final_name].predict(X_test)))  # scored once
pd.DataFrame(rows).round(4).to_csv("outputs/part_b_metrics.csv", index=False)
print("val ranking:", ranking, "| best depth:", best_depth, "| final:", final_name)
