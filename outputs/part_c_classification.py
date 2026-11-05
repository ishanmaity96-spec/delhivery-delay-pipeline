"""Part C - delay classifier, feature signal, tuning, alert threshold."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"]

# 1. stratified split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print("sizes", len(X_train), len(X_val), len(X_test), "| delayed", y_train.sum(), y_val.sum(), y_test.sum())


def cm_row(name, y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0)}


# 2. blind baseline and three classifiers (0.5 rule)
model_rows = []
r = cm_row("blind", y_val, np.zeros(len(y_val), dtype=int)); r["precision"] = 0
model_rows.append({"model": "blind_all_on_time", **r})

logit = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)
model_rows.append({"model": "logistic_regression", **cm_row("lr", y_val, logit.predict(X_val))})

best_d, best_f1 = None, -1
for d in [2, 3, 4, 5, 6, 8, 10]:          # ascending, strict > keeps the smaller d on ties
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    f = f1_score(y_val, t.predict(X_val))
    if f > best_f1 + 1e-12:
        best_d, best_f1 = d, f
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_train, y_train)
model_rows.append({"model": f"decision_tree_depth_{best_d}", **cm_row("tree", y_val, tree.predict(X_val))})

rf_default = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
model_rows.append({"model": "random_forest_default", **cm_row("rf", y_val, rf_default.predict(X_val))})
pd.DataFrame(model_rows).round(4).to_csv("outputs/part_c_models.csv", index=False)

# 3. which columns carry signal
rf17 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
skb = SelectKBest(f_classif, k="all").fit(X_train, y_train)
fs = pd.DataFrame({"feature": FEATURES, "f_score": skb.scores_, "rf_importance": rf17.feature_importances_})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="first").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs = fs[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].sort_values("f_rank")
fs.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)

k_rows = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("sel", SelectKBest(f_classif, k=k)),
                  ("rf", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    k_rows.append({"k": k, "val_f1": f1_score(y_val, p.predict(X_val))})
pd.DataFrame(k_rows).round(4).to_csv("outputs/part_c_selectk.csv", index=False)

# 4. tuning (k = 7, selector fitted on training rows only)
sel7 = SelectKBest(f_classif, k=7).fit(X_train, y_train)
Xtr7, Xva7 = sel7.transform(X_train), sel7.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(Xtr7, y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5,
                        random_state=42).fit(Xtr7, y_train)
rf7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr7, y_train)


def params_str(p):
    return str({k: int(v) for k, v in p.items()})


tuning = pd.DataFrame([
    {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 27 * 5,
     "best_params": params_str(gs.best_params_), "best_cv_accuracy": gs.best_score_,
     "val_f1": f1_score(y_val, gs.best_estimator_.predict(Xva7)), "_p": gs.best_params_},
    {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 9 * 5,
     "best_params": params_str(rs.best_params_), "best_cv_accuracy": rs.best_score_,
     "val_f1": f1_score(y_val, rs.best_estimator_.predict(Xva7)), "_p": rs.best_params_},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": "{'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1}",
     "best_cv_accuracy": np.nan, "val_f1": f1_score(y_val, rf7.predict(Xva7)),
     "_p": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}},
])
tuning.drop(columns="_p").round(4).to_csv("outputs/part_c_tuning.csv", index=False)

# 5. final configuration: highest val_f1; ties -> default, grid, randomized
prio = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
tuning["_r"] = tuning["val_f1"].round(10)
win = sorted(range(3), key=lambda i: (-tuning.loc[i, "_r"], prio[tuning.loc[i, "search"]]))[0]
chosen_name, cp = tuning.loc[win, "search"], tuning.loc[win, "_p"]
final = Pipeline([("selector", SelectKBest(f_classif, k=7)),
                  ("model", RandomForestClassifier(random_state=42, **cp))]).fit(X_train, y_train)

# 6. alert threshold from validation probabilities
pv = final.predict_proba(X_val)[:, 1]
thr_rows = []
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    thr_rows.append({"threshold": t, **cm_row("t", y_val, (pv >= t).astype(int))})
thr = pd.DataFrame(thr_rows)
thr.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)
ok = thr[thr["recall"] >= 0.80]
chosen_t = float(ok["threshold"].max()) if len(ok) else 0.1

# 7. one test evaluation
pt = final.predict_proba(X_test)[:, 1]
test_row = {"threshold": chosen_t, **cm_row("test", y_test, (pt >= chosen_t).astype(int))}
pd.DataFrame([test_row]).round(4).to_csv("outputs/part_c_final_test.csv", index=False)
pd.DataFrame([{"selector_k": 7, "n_estimators": int(cp["n_estimators"]),
               "max_depth": "unlimited" if cp["max_depth"] is None else int(cp["max_depth"]),
               "min_samples_leaf": int(cp["min_samples_leaf"]), "threshold": chosen_t,
               "chosen_from": chosen_name}]).to_csv("outputs/part_c_final_config.csv", index=False)
print("chosen:", chosen_name, cp, "threshold", chosen_t)
