"""
upay Shield - Feature Engineering
-----------------------------------
Turns raw transactions into the signals the risk model actually looks at.
Every feature here is something a human analyst could also check by hand --
that's deliberate, it's what makes the model's decisions explainable later.

Features built:
  sender side (is this normal FOR THIS PERSON?):
    - is_new_recipient        : sender has never sent to this receiver before
    - amount_vs_avg_ratio     : this amount vs. the sender's own historical average
    - amount_zscore           : how many std-devs this amount is from the sender's norm
    - unusual_hour            : transaction hour outside the sender's normal active hours
    - sender_velocity_10min   : how many transactions this sender made in the last 10 min
    - new_device_for_sender   : device_id not seen before for this sender
    - cross_district          : sender's declared district differs from their usual one

  receiver side (does the money's DESTINATION look like a mule wallet?):
    - receiver_age_days       : days since this wallet's first-ever appearance in the data
    - receiver_indegree_1h    : distinct senders to this receiver in the last 1 hour
    - receiver_fanout_30min   : whether receiver sends most of it back out within 30 min

Output: features.csv (one row per transaction, ready for model training)
"""

import pandas as pd
import numpy as np


def build_features(txns: pd.DataFrame) -> pd.DataFrame:
    df = txns.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ---- sender history state (updated as we walk through time) ----
    seen_pairs = set()                 # (sender, receiver) seen before
    sender_amounts = {}                # sender -> list of past amounts
    sender_devices = {}                # sender -> set of devices seen
    sender_recent_times = {}           # sender -> list of recent timestamps (for velocity)
    receiver_first_seen = {}           # receiver -> first timestamp seen
    receiver_recent_senders = {}       # receiver -> list of (timestamp, sender) for fan-in

    is_new_recipient = np.zeros(len(df), dtype=int)
    amount_vs_avg_ratio = np.zeros(len(df))
    amount_zscore = np.zeros(len(df))
    unusual_hour = np.zeros(len(df), dtype=int)
    sender_velocity_10min = np.zeros(len(df), dtype=int)
    new_device_for_sender = np.zeros(len(df), dtype=int)
    receiver_age_days = np.zeros(len(df))
    receiver_indegree_1h = np.zeros(len(df), dtype=int)

    for i, row in df.iterrows():
        s, r, amt, ts = row["sender_id"], row["receiver_id"], row["amount"], row["timestamp"]
        hour = ts.hour

        # new recipient?
        pair = (s, r)
        is_new_recipient[i] = 0 if pair in seen_pairs else 1
        seen_pairs.add(pair)

        # amount vs sender's own history
        hist = sender_amounts.get(s, [])
        if hist:
            mean = np.mean(hist)
            std = np.std(hist) if len(hist) > 1 else max(mean * 0.3, 1.0)
            amount_vs_avg_ratio[i] = amt / max(mean, 1.0)
            amount_zscore[i] = (amt - mean) / max(std, 1.0)
        else:
            amount_vs_avg_ratio[i] = 1.0
            amount_zscore[i] = 0.0
        sender_amounts.setdefault(s, []).append(amt)

        # unusual hour: flag late-night (11pm-6am) as a simple, explainable proxy
        # for "outside this person's normal pattern" without needing a huge history
        unusual_hour[i] = 1 if (hour >= 23 or hour <= 5) else 0

        # device novelty
        devs = sender_devices.setdefault(s, set())
        new_device_for_sender[i] = 0 if (row["device_id"] in devs or len(devs) == 0) else 1
        devs.add(row["device_id"])

        # sender velocity: txns by this sender in the last 10 minutes
        recent = sender_recent_times.setdefault(s, [])
        recent = [t for t in recent if (ts - t).total_seconds() <= 600]
        sender_velocity_10min[i] = len(recent)
        recent.append(ts)
        sender_recent_times[s] = recent

        # receiver wallet age
        if r not in receiver_first_seen:
            receiver_first_seen[r] = ts
        receiver_age_days[i] = (ts - receiver_first_seen[r]).total_seconds() / 86400.0

        # receiver fan-in: distinct senders to this receiver in the last hour
        rlist = receiver_recent_senders.setdefault(r, [])
        rlist = [(t, sd) for (t, sd) in rlist if (ts - t).total_seconds() <= 3600]
        distinct_senders = len(set(sd for _, sd in rlist))
        receiver_indegree_1h[i] = distinct_senders
        rlist.append((ts, s))
        receiver_recent_senders[r] = rlist

    df["is_new_recipient"] = is_new_recipient
    df["amount_vs_avg_ratio"] = amount_vs_avg_ratio
    df["amount_zscore"] = amount_zscore
    df["unusual_hour"] = unusual_hour
    df["sender_velocity_10min"] = sender_velocity_10min
    df["new_device_for_sender"] = new_device_for_sender
    # bucket wallet age into a few bands rather than a raw day-count.
    # Reason: a raw continuous value taught the model an overly sharp rule
    # ("age exactly 0.0" vs "0.2" looked totally different to it, even
    # though both just mean 'brand new wallet'). Bucketing groups nearby,
    # equally-risky ages together so the signal is robust, not brittle:
    #   0 = under 1 day old   1 = under 1 week   2 = under 1 month   3 = older
    df["receiver_age_days"] = pd.cut(
        receiver_age_days, bins=[-0.001, 1, 7, 30, 1e9], labels=[0, 1, 2, 3]
    ).astype(int)
    df["receiver_indegree_1h"] = receiver_indegree_1h

    # receiver fan-out within 30 min: did the receiver send most of what they
    # just received back out again quickly? (classic mule pass-through)
    df = _add_fanout_feature(df)

    return df


def _add_fanout_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    out_events = df[["timestamp", "sender_id", "amount"]].rename(
        columns={"sender_id": "wallet", "amount": "out_amount", "timestamp": "out_ts"}
    ).sort_values("out_ts")

    fanout_flag = np.zeros(len(df), dtype=int)
    out_by_wallet = {}
    for w, g in out_events.groupby("wallet"):
        out_by_wallet[w] = g[["out_ts", "out_amount"]].values.tolist()

    for i, row in df.iterrows():
        r, ts, amt = row["receiver_id"], row["timestamp"], row["amount"]
        events = out_by_wallet.get(r, [])
        total_out_30min = sum(
            oa for (ots, oa) in events
            if 0 <= (pd.Timestamp(ots) - ts).total_seconds() <= 1800
        )
        fanout_flag[i] = 1 if total_out_30min >= 0.7 * amt and amt > 0 else 0

    df["receiver_fanout_30min"] = fanout_flag
    return df


FEATURE_COLUMNS = [
    "amount",
    "is_new_recipient",
    "amount_vs_avg_ratio",
    "amount_zscore",
    "unusual_hour",
    "sender_velocity_10min",
    "new_device_for_sender",
    "receiver_age_days",
    "receiver_indegree_1h",
    "receiver_fanout_30min",
]


if __name__ == "__main__":
    import os
    txns = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "..", "data", "transactions.csv"))
    feats = build_features(txns)
    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "features.csv")
    feats.to_csv(out_path, index=False)
    print(f"Wrote {len(feats)} rows with features to {out_path}")
    print(feats[FEATURE_COLUMNS + ["is_scam"]].groupby("is_scam").mean().T)
