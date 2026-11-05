"""Score new parcels and write the action list. Re-uses the shared functions in src/shipment_prep.py."""
import joblib
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

new = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute = pd.read_csv("models/impute_values.csv")
medians = {r.column: float(r.median) for r in impute.itertuples(index=False)}
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

clean, _ = clean_shipments(new, medians=medians)
feats = build_features(clean, hubs)
feats["delay_probability"] = model.predict_proba(feats[order])[:, 1].round(4)
feats["flagged"] = (feats["delay_probability"] >= threshold).astype(int)
feats = feats.merge(segments, on="hub_id", how="left")
feats["action"] = "No action"
feats.loc[feats["flagged"] == 1, "action"] = "Proactive customer call"
feats.loc[(feats["flagged"] == 1) & (feats["segment"] == "Escalate"), "action"] = "Escalate to hub manager"
feats[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]].to_csv(
    "outputs/new_shipment_actions.csv", index=False)
print("scored", len(feats), "parcels")
