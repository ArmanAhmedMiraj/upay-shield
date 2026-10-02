"""
upay Shield - Mule Network Discovery
----------------------------------------
Builds a directed transaction graph (wallet -> wallet, edge = transfer) and
looks for the shape a money-mule wallet makes: lots of money flowing IN from
many unrelated people in a short window, then flowing back OUT again fast.

This is deliberately simple and explainable (not a black-box GNN) so an
analyst can see exactly why a wallet got flagged: "12 distinct senders in an
hour, 3-day-old wallet, 91% of the money cashed out within 2 hours."

Output: mule_report.json - ranked list of suspicious wallets + their evidence
"""

import json
import os
import pandas as pd
import networkx as nx
from collections import defaultdict

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "..", "data")


def build_graph(txns: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in txns.iterrows():
        s, r, amt, ts = row["sender_id"], row["receiver_id"], row["amount"], row["timestamp"]
        if G.has_edge(s, r):
            G[s][r]["weight"] += amt
            G[s][r]["count"] += 1
        else:
            G.add_edge(s, r, weight=amt, count=1)
    return G


def wallet_evidence(txns: pd.DataFrame, wallet: str, dataset_end) -> dict:
    """Compute the human-readable evidence for one wallet: fan-in burst,
    wallet age, and how fast money leaves again after arriving."""
    txns = txns.copy()
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])

    incoming = txns[txns["receiver_id"] == wallet].sort_values("timestamp")
    outgoing = txns[txns["sender_id"] == wallet].sort_values("timestamp")

    if incoming.empty:
        return None

    first_seen = incoming["timestamp"].min()
    wallet_age_days_now = (dataset_end - first_seen).total_seconds() / 86400.0
    distinct_senders = incoming["sender_id"].nunique()
    total_in = incoming["amount"].sum()
    total_out = outgoing["amount"].sum()
    cash_out_ratio = round(total_out / total_in, 3) if total_in > 0 else 0.0

    # tightest 60-minute window: max distinct senders within any 1-hour span
    max_fanin_1h = 0
    times = incoming["timestamp"].tolist()
    for t in times:
        window = incoming[(incoming["timestamp"] >= t) &
                           (incoming["timestamp"] <= t + pd.Timedelta(hours=1))]
        max_fanin_1h = max(max_fanin_1h, window["sender_id"].nunique())

    # time from last big inflow to first big outflow after it (pass-through speed)
    pass_through_minutes = None
    if not outgoing.empty:
        last_in = incoming["timestamp"].max()
        later_out = outgoing[outgoing["timestamp"] >= last_in]
        if not later_out.empty:
            pass_through_minutes = round(
                (later_out["timestamp"].min() - last_in).total_seconds() / 60, 1
            )

    return {
        "wallet": wallet,
        "distinct_senders_total": int(distinct_senders),
        "max_distinct_senders_in_1h": int(max_fanin_1h),
        "total_received": round(float(total_in), 2),
        "total_sent_out": round(float(total_out), 2),
        "cash_out_ratio": cash_out_ratio,
        "first_seen": first_seen.isoformat(),
        "wallet_age_days": round(wallet_age_days_now, 2),
        "pass_through_minutes_after_last_inflow": pass_through_minutes,
    }


def score_mule_likelihood(evidence: dict) -> float:
    """Explainable rule-based score (0-1) combining the evidence.

    Important design choice: a fast-fan-in burst (>=3 distinct senders in
    one hour) is a GATE, not just a point score. Busy agents and regular
    users naturally recycle money (high cash-out ratio over their whole
    history) -- that alone is NOT suspicious, it's how agents work. What's
    suspicious is money arriving from many strangers in a short burst AND
    leaving again fast, on a wallet with little history. Each clause below
    mirrors something an analyst would actually check by hand.
    """
    reasons = []

    fanin = evidence["max_distinct_senders_in_1h"]
    if fanin < 3:
        # no burst fan-in at all -> not mule-pattern, regardless of other stats
        evidence["mule_score"] = 0.0
        evidence["reasons"] = []
        return evidence

    score = 0.0
    if fanin >= 8:
        score += 0.40
        reasons.append(f"{fanin} distinct senders within a single hour (very high burst)")
    else:
        score += 0.20
        reasons.append(f"{fanin} distinct senders within a single hour")

    if evidence["wallet_age_days"] <= 7:
        score += 0.25
        reasons.append(f"wallet first appeared only {evidence['wallet_age_days']:.1f} days ago")
    elif evidence["wallet_age_days"] <= 30:
        score += 0.10
        reasons.append(f"relatively new wallet ({evidence['wallet_age_days']:.0f} days of history)")

    pt = evidence["pass_through_minutes_after_last_inflow"]
    if pt is not None and pt <= 180 and evidence["cash_out_ratio"] >= 0.7:
        score += 0.35
        reasons.append(
            f"{evidence['cash_out_ratio']*100:.0f}% of received funds moved back out "
            f"within {pt:.0f} minutes of the last incoming transfer"
        )
    elif evidence["cash_out_ratio"] >= 0.7:
        score += 0.10
        reasons.append(f"{evidence['cash_out_ratio']*100:.0f}% of received funds moved back out (over time)")

    score = min(score, 1.0)
    evidence["mule_score"] = round(score, 3)
    evidence["reasons"] = reasons
    return evidence


def main():
    txns = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])

    G = build_graph(txns)
    print(f"Graph built: {G.number_of_nodes()} wallets, {G.number_of_edges()} edges")

    dataset_end = txns["timestamp"].max()

    # candidate wallets: anyone who received from >=3 distinct senders ever,
    # EXCLUDING agents -- agents are a verified, licensed entity type that
    # naturally does high-volume pass-through (that's their job). Flagging
    # agents here would just be noise; agent-specific risk (an agent working
    # WITH a mule ring) is a different, separate check, not this one.
    in_degree_counts = defaultdict(set)
    receiver_types = {}
    for _, row in txns.iterrows():
        in_degree_counts[row["receiver_id"]].add(row["sender_id"])
        receiver_types[row["receiver_id"]] = row["receiver_type"]

    candidates = [
        w for w, senders in in_degree_counts.items()
        if len(senders) >= 3 and receiver_types.get(w) != "agent"
    ]
    print(f"Candidate wallets to evaluate (>=3 distinct senders ever, excluding agents): {len(candidates)}")

    results = []
    for w in candidates:
        ev = wallet_evidence(txns, w, dataset_end)
        if ev is None:
            continue
        ev = score_mule_likelihood(ev)
        if ev["mule_score"] >= 0.4:
            results.append(ev)

    results = sorted(results, key=lambda r: -r["mule_score"])

    # also find connected clusters among the flagged wallets + their
    # immediate senders, to show "this isn't one wallet, it's a ring"
    flagged = set(r["wallet"] for r in results[:50])
    sub_edges = [(u, v) for u, v in G.edges() if v in flagged]
    H = nx.Graph()
    H.add_edges_from(sub_edges)
    components = [list(c) for c in nx.connected_components(H) if len(c) >= 4]

    report = {
        "graph_summary": {
            "total_wallets": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "candidate_wallets_checked": len(candidates),
            "flagged_wallets": len(results),
        },
        "top_suspicious_wallets": results[:30],
        "suspicious_clusters": [
            {"cluster_id": i, "size": len(c), "wallets": c[:25]}
            for i, c in enumerate(components)
        ],
    }

    with open(os.path.join(DATA_DIR, "mule_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFlagged {len(results)} wallets with mule-like behavior.")
    print(f"Found {len(components)} suspicious connected clusters (size >= 4).")
    print(f"Saved mule_report.json to {DATA_DIR}")
    if results:
        print("\nTop 5 most suspicious wallets:")
        for r in results[:5]:
            print(f"  {r['wallet']}: score={r['mule_score']} | {', '.join(r['reasons'])}")


if __name__ == "__main__":
    main()
