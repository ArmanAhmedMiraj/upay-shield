"""
upay Shield - Backend API
-----------------------------
Flask app exposing the risk engine + mule network + investigation assistant
as a simple REST API for the analyst dashboard (and, later, the customer
app's scam-moment check).

Endpoints:
  GET  /api/health
  GET  /api/stats                         - summary numbers for the dashboard header
  GET  /api/transactions?risk=high&limit=50
  GET  /api/transactions/<txn_id>         - full detail + AI case summary
  POST /api/transactions/<txn_id>/decision  {"decision": "hold"|"approve"|"dismiss"}
  GET  /api/mule/wallets                  - ranked suspicious wallets
  GET  /api/mule/clusters                 - suspicious connected clusters
  POST /api/score                         - score a NEW transaction on the fly
                                             (what the real-time check would call)

Run:  python3 api.py   (serves on http://localhost:5000)
"""

import os
import json
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from flask import send_from_directory

from features import FEATURE_COLUMNS
from investigation_assistant import template_explain, llm_explain, _guess_scam_type

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "..", "data")

app = Flask(__name__)

# ---- load everything once at startup ----
print("Loading data and model...")
scored_df = pd.read_csv(os.path.join(DATA_DIR, "scored_transactions.csv"))
with open(os.path.join(DATA_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(DATA_DIR, "mule_report.json")) as f:
    mule_report = json.load(f)
with open(os.path.join(DATA_DIR, "evaluation.json")) as f:
    evaluation = json.load(f)
with open(os.path.join(DATA_DIR, "feature_importance.json")) as f:
    feature_importance = json.load(f)

mule_by_wallet = {w["wallet"]: w for w in mule_report["top_suspicious_wallets"]}

# in-memory analyst decisions (demo only -- would be a real table in production)
decisions = {}


def risk_level(score):
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def txn_to_dict(row):
    feats = {c: (float(row[c]) if c in row else None) for c in FEATURE_COLUMNS}
    mule_evidence = mule_by_wallet.get(row["receiver_id"])
    scam_guess = _guess_scam_type({"features": feats, "mule_evidence": mule_evidence})
    return {
        "txn_id": row["txn_id"],
        "timestamp": row["timestamp"],
        "sender_id": row["sender_id"],
        "receiver_id": row["receiver_id"],
        "channel": row["channel"],
        "amount": float(row["amount"]),
        "sender_district": row.get("sender_district"),
        "risk_score": float(row["risk_score"]),
        "risk_level": risk_level(row["risk_score"]),
        "features": feats,
        "decision": decisions.get(row["txn_id"], "pending"),
        # lightweight guess for list views; the full grounded explanation
        # (template or LLM) is computed on-demand in the detail endpoint
        "ai_case_summary": {"scam_type_guess": scam_guess},
    }


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "transactions_loaded": len(scored_df)})


@app.route("/api/stats")
def stats():
    total = len(scored_df)
    flagged = scored_df[scored_df["risk_score"] >= 0.4]
    high = scored_df[scored_df["risk_score"] >= 0.7]
    return jsonify({
        "total_transactions": int(total),
        "flagged_medium_plus": int(len(flagged)),
        "flagged_high": int(len(high)),
        "mule_wallets_detected": mule_report["graph_summary"]["flagged_wallets"],
        "suspicious_clusters": len(mule_report["suspicious_clusters"]),
        "model_evaluation": evaluation,
        "feature_importance": feature_importance,
    })


@app.route("/api/transactions")
def list_transactions():
    risk = request.args.get("risk", "all")
    limit = int(request.args.get("limit", 50))

    df = scored_df.copy()
    if risk == "high":
        df = df[df["risk_score"] >= 0.7]
    elif risk == "medium":
        df = df[(df["risk_score"] >= 0.4) & (df["risk_score"] < 0.7)]
    elif risk == "flagged":
        df = df[df["risk_score"] >= 0.4]

    df = df.sort_values("risk_score", ascending=False).head(limit)
    return jsonify([txn_to_dict(row) for _, row in df.iterrows()])


@app.route("/api/transactions/<txn_id>")
def transaction_detail(txn_id):
    row = scored_df[scored_df["txn_id"] == txn_id]
    if row.empty:
        return jsonify({"error": "not found"}), 404
    row = row.iloc[0]
    txn = txn_to_dict(row)

    mule_evidence = mule_by_wallet.get(row["receiver_id"])
    explanation = llm_explain(txn, row["risk_score"], mule_evidence)

    txn["ai_case_summary"] = explanation
    txn["mule_network_evidence"] = mule_evidence
    return jsonify(txn)


@app.route("/api/transactions/<txn_id>/decision", methods=["POST"])
def set_decision(txn_id):
    body = request.get_json(force=True, silent=True) or {}
    decision = body.get("decision")
    if decision not in ("approve", "hold", "dismiss"):
        return jsonify({"error": "decision must be approve|hold|dismiss"}), 400
    decisions[txn_id] = decision
    return jsonify({"txn_id": txn_id, "decision": decision})


@app.route("/api/mule/wallets")
def mule_wallets():
    return jsonify(mule_report["top_suspicious_wallets"])


@app.route("/api/mule/clusters")
def mule_clusters():
    return jsonify(mule_report["suspicious_clusters"])


@app.route("/api/score", methods=["POST"])
def score_new_transaction():
    """Score a brand-new transaction on the fly -- this is the shape of
    call the customer app's real-time 'scam-moment check' would make
    before letting a transaction go through."""
    body = request.get_json(force=True, silent=True) or {}
    required = FEATURE_COLUMNS
    missing = [f for f in required if f not in body]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400

    X = pd.DataFrame([{f: body[f] for f in required}])
    score = float(model.predict_proba(X)[:, 1][0])
    level = risk_level(score)

    action = {
        "high": "hold_for_review",
        "medium": "require_step_up_confirmation",
        "low": "allow",
    }[level]

    return jsonify({
        "risk_score": round(score, 4),
        "risk_level": level,
        "recommended_action": action,
    })


FRONTEND_DIR = os.path.join(HERE, "..", "..", "frontend")


@app.route("/")
def serve_dashboard():
    return send_from_directory(FRONTEND_DIR, "dashboard.html")


@app.route("/<path:filename>")
def serve_frontend_asset(filename):
    # serves demo_data.js and any other static frontend file alongside the dashboard
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"upay Shield API + dashboard running on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
