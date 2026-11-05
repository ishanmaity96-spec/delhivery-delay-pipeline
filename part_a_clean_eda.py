"""Part A - cleaning, modelling table, checks, delay evidence, correlations."""
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# "before" measurements: after de-duplication, before any filling
dedup = raw.drop_duplicates()
decl_before = pd.to_numeric(dedup["declared_value"].astype("string").str.replace("\u20b9", "", regex=False)
                            .str.replace(",", "", regex=False), errors="coerce")
sentinel = int((dedup["parcel_weight_kg"] > 100).sum())
missing_weight_before = int(dedup["parcel_weight_kg"].isna().sum())

clean, medians = clean_shipments(raw)
table = build_features(clean, hubs)
table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
table.to_csv("data/modelling_table.csv", index=False)

q1, q3 = table["delivery_hours"].quantile(0.25), table["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
log_v = table["log_declared_value"]

checks = {
    "raw_rows": len(raw),
    "duplicate_rows_removed": len(raw) - len(dedup),
    "rows_after_dedup": len(dedup),
    "missing_distance_km_before": int(dedup["distance_km"].isna().sum()),
    "missing_parcel_weight_before": missing_weight_before,
    "sentinel_weight_rows": sentinel,
    "missing_declared_value_before": int(decl_before.isna().sum()),
    "missing_payment_mode_before": int(dedup["payment_mode"].isna().sum()),
    "service_type_spellings_before": int(dedup["service_type"].nunique()),
    "service_type_spellings_after": int(clean["service_type"].nunique()),
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "rows_after_hub_merge": len(table),
    "nulls_in_features_and_target": int(table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
    "delayed_count": int(table["is_delayed"].sum()),
    "delay_rate": table["is_delayed"].mean(),
    "hours_q1": q1, "hours_q3": q3,
    "hours_lower_bound": lo, "hours_upper_bound": hi,
    "hours_outlier_count": int(((table["delivery_hours"] < lo) | (table["delivery_hours"] > hi)).sum()),
    "declared_value_mean_inr": table["declared_value_inr"].mean(),
    "declared_value_median_inr": table["declared_value_inr"].median(),
    "log_declared_value_mean": log_v.mean(),
    "log_declared_value_median": log_v.median(),
}
# integers stay integers; decimals are rounded to 4 places only when written
fmt = lambda v: str(int(v)) if float(v).is_integer() and not isinstance(v, float) else f"{round(float(v), 4):g}"
pd.DataFrame({"metric": list(checks), "value": [fmt(v) for v in checks.values()]}).to_csv(
    "outputs/part_a_checks.csv", index=False)

# delay evidence table
table["distance_band"] = pd.cut(table["distance_km"], bins=[0, 200, 400, 600, 2000],
                                labels=["0-200", "201-400", "401-600", "601+"], right=True)
rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = table.groupby(dim, observed=True)
    for key, sub in g:
        rows.append({"dimension": dim, "group": str(key), "shipments": len(sub),
                     "delayed": int(sub["is_delayed"].sum()), "delay_rate": sub["is_delayed"].mean(),
                     "mean_hours": sub["delivery_hours"].mean()})
pd.DataFrame(rows).round(4).to_csv("outputs/part_a_delay_by_group.csv", index=False)

# correlations of numeric columns
corr = table.drop(columns=["distance_band"]).select_dtypes(include="number").corr()
corr.round(4).to_csv("outputs/part_a_correlations.csv")
print("Part A done:", len(table), "rows")
