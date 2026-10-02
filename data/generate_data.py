"""
upay Shield - Synthetic Data Generator
----------------------------------------
Generates realistic-but-fake MFS (mobile financial service) data:
  - users.csv       : customer wallets
  - agents.csv       : cash-in/cash-out agents
  - transactions.csv : transactions, with normal behavior PLUS three
                        injected labeled patterns so the model has
                        something real to learn:
                          1) social_engineering  - scam-call style transfer
                          2) mule_ring            - fan-in then fast cash-out
                          3) account_takeover     - new device/location burst

No real personal data is used anywhere. All names/numbers are fake.
Run:  python3 generate_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import uuid

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

N_USERS = 2000
N_AGENTS = 120
N_DAYS = 60
SIM_START = datetime(2026, 8, 1)

DISTRICTS = [
    "Dhaka", "Chattogram", "Khulna", "Rajshahi", "Sylhet",
    "Barishal", "Rangpur", "Mymensingh", "Comilla", "Narayanganj",
]

FIRST_NAMES = ["Rahim", "Karim", "Fatema", "Ayesha", "Jahid", "Nasrin", "Sabbir",
               "Mitu", "Rafiq", "Shirin", "Tanvir", "Lima", "Imran", "Nabila",
               "Hasan", "Ruma", "Shakib", "Dolly", "Masud", "Shila"]
LAST_NAMES = ["Islam", "Ahmed", "Begum", "Khan", "Hossain", "Akter", "Chowdhury",
              "Rahman", "Uddin", "Sultana"]


def fake_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def make_users(n):
    rows = []
    for i in range(n):
        uid = f"U{i:05d}"
        join_offset = random.randint(0, 500)  # days before sim start, account age
        join_date = SIM_START - timedelta(days=join_offset)
        # a user's "typical" transfer amount (taka) - log-normal, so most are
        # small-to-medium with a long tail of bigger spenders/traders
        typical_amount = float(np.round(np.random.lognormal(mean=6.5, sigma=0.9), 2))
        typical_amount = max(50.0, min(typical_amount, 60000.0))
        # how active during the day: most people cluster around daytime hours
        active_hours = sorted(random.sample(range(7, 22), k=random.randint(4, 8)))
        role = np.random.choice(["customer", "merchant", "vendor"], p=[0.82, 0.10, 0.08])
        rows.append({
            "user_id": uid,
            "name": fake_name(),
            "district": random.choice(DISTRICTS),
            "join_date": join_date.date().isoformat(),
            "role": role,
            "typical_amount": typical_amount,
            "active_hours": ",".join(map(str, active_hours)),
            "device_id": f"D{uuid.uuid4().hex[:8]}",
        })
    return pd.DataFrame(rows)


def make_agents(n):
    rows = []
    for i in range(n):
        aid = f"A{i:04d}"
        rows.append({
            "agent_id": aid,
            "name": fake_name() + " Store",
            "district": random.choice(DISTRICTS),
        })
    return pd.DataFrame(rows)


def build_social_graph(users_df, avg_contacts=9):
    """Each user has a small set of regular contacts they transact with
    repeatedly -- this is what 'normal' looks like."""
    ids = users_df["user_id"].tolist()
    graph = {}
    for uid in ids:
        k = max(2, int(np.random.poisson(avg_contacts)))
        contacts = random.sample([x for x in ids if x != uid], k=min(k, len(ids) - 1))
        graph[uid] = contacts
    return graph


def gen_normal_transactions(users_df, agents_df, social_graph, n_days):
    """Everyday send-money, cash-in and cash-out activity between a user
    and their regular contacts / agents."""
    rows = []
    users = users_df.set_index("user_id")
    agent_ids = agents_df["agent_id"].tolist()
    txn_counter = 0

    for day in range(n_days):
        date = SIM_START + timedelta(days=day)
        for uid, row in users_df.iterrows():
            u = row["user_id"]
            # not everyone transacts every day
            if random.random() > 0.35:
                continue
            n_txns_today = np.random.poisson(1.3) + 1
            for _ in range(n_txns_today):
                channel = np.random.choice(
                    ["send_money", "cash_out", "cash_in", "payment"],
                    p=[0.45, 0.25, 0.15, 0.15],
                )
                hour_choices = [int(h) for h in users.loc[u, "active_hours"].split(",")]
                hour = random.choice(hour_choices)
                minute = random.randint(0, 59)
                ts = date + timedelta(hours=hour, minutes=minute)

                amount = float(np.round(
                    np.random.lognormal(mean=np.log(max(users.loc[u, "typical_amount"], 50)),
                                         sigma=0.35), 2))
                amount = max(20.0, min(amount, 150000.0))

                if channel in ("send_money", "payment"):
                    contacts = social_graph.get(u, [])
                    if contacts and random.random() < 0.88:
                        receiver = random.choice(contacts)
                    else:
                        receiver = random.choice(users_df["user_id"].tolist())
                    receiver_type = "user"
                elif channel == "cash_out":
                    receiver = random.choice(agent_ids)
                    receiver_type = "agent"
                else:  # cash_in
                    receiver = u
                    u_actual_sender = random.choice(agent_ids)
                    receiver = u
                    rows.append({
                        "txn_id": f"T{txn_counter:07d}", "timestamp": ts.isoformat(),
                        "sender_id": u_actual_sender, "sender_type": "agent",
                        "receiver_id": u, "receiver_type": "user",
                        "channel": "cash_in", "amount": amount,
                        "sender_district": users.loc[u, "district"],
                        "device_id": users.loc[u, "device_id"],
                        "is_scam": 0, "scam_type": "none",
                    })
                    txn_counter += 1
                    continue

                rows.append({
                    "txn_id": f"T{txn_counter:07d}", "timestamp": ts.isoformat(),
                    "sender_id": u, "sender_type": "user",
                    "receiver_id": receiver, "receiver_type": receiver_type,
                    "channel": channel, "amount": amount,
                    "sender_district": users.loc[u, "district"],
                    "device_id": users.loc[u, "device_id"],
                    "is_scam": 0, "scam_type": "none",
                })
                txn_counter += 1
    return pd.DataFrame(rows), txn_counter


def inject_social_engineering(users_df, txn_counter, n_cases=140):
    """Scam-call pattern: victim sends an unusually large amount to a
    recipient they've never paid before, often at an odd hour, often a
    big share of what they normally send in one go.

    ~35% of scam wallets collect from 2-3 victims over a few days before
    going quiet (more realistic, and gives the model varied receiver-age
    examples instead of only ever seeing "wallet age = 0", which would
    teach it a too-narrow, brittle rule)."""
    rows = []
    victims = users_df.sample(n_cases, random_state=1).to_dict("records")
    i = 0
    while i < len(victims):
        reused = random.random() < 0.35 and i + 1 < len(victims)
        group_size = random.randint(2, 3) if reused else 1
        group = victims[i:i + group_size]
        i += group_size

        scam_wallet = f"M{uuid.uuid4().hex[:6]}"
        start_day = random.randint(10, N_DAYS - 5)
        for offset, v in enumerate(group):
            day = start_day + (offset * random.randint(0, 3) if reused else 0)
            ts = SIM_START + timedelta(days=day, hours=random.choice([12, 13, 20, 21, 22]),
                                        minutes=random.randint(0, 59))
            amount = float(np.round(v["typical_amount"] * random.uniform(4, 15), 2))
            amount = min(amount, 150000.0)
            rows.append({
                "txn_id": f"T{txn_counter:07d}", "timestamp": ts.isoformat(),
                "sender_id": v["user_id"], "sender_type": "user",
                "receiver_id": scam_wallet, "receiver_type": "user",
                "channel": "send_money", "amount": amount,
                "sender_district": v["district"], "device_id": v["device_id"],
                "is_scam": 1, "scam_type": "social_engineering",
            })
            txn_counter += 1
    return pd.DataFrame(rows), txn_counter, list(set(r["receiver_id"] for r in rows))


def inject_mule_rings(users_df, txn_counter, n_rings=6, senders_per_ring=14):
    """Fan-in / fast fan-out pattern: a freshly created wallet receives
    money from many unrelated senders within a short window, then
    cashes most of it out within minutes."""
    rows = []
    mule_wallets = []
    for ring in range(n_rings):
        mule_id = f"M{uuid.uuid4().hex[:6]}"
        mule_wallets.append(mule_id)
        day = random.randint(5, N_DAYS - 2)
        base_ts = SIM_START + timedelta(days=day, hours=random.randint(9, 20))
        senders = users_df.sample(senders_per_ring, random_state=ring + 100)
        total_in = 0.0
        for i, (_, s) in enumerate(senders.iterrows()):
            ts = base_ts + timedelta(minutes=random.randint(0, 90))
            amount = float(np.round(random.uniform(1500, 20000), 2))
            total_in += amount
            rows.append({
                "txn_id": f"T{txn_counter:07d}", "timestamp": ts.isoformat(),
                "sender_id": s["user_id"], "sender_type": "user",
                "receiver_id": mule_id, "receiver_type": "user",
                "channel": "send_money", "amount": amount,
                "sender_district": s["district"], "device_id": s["device_id"],
                "is_scam": 1, "scam_type": "mule_ring",
            })
            txn_counter += 1
        # fast cash-out of most of the money, shortly after the fan-in window
        cash_out_ts = base_ts + timedelta(minutes=random.randint(95, 150))
        rows.append({
            "txn_id": f"T{txn_counter:07d}", "timestamp": cash_out_ts.isoformat(),
            "sender_id": mule_id, "sender_type": "user",
            "receiver_id": f"A{random.randint(0, N_AGENTS - 1):04d}", "receiver_type": "agent",
            "channel": "cash_out", "amount": round(total_in * random.uniform(0.85, 0.97), 2),
            "sender_district": random.choice(DISTRICTS), "device_id": f"D{uuid.uuid4().hex[:8]}",
            "is_scam": 1, "scam_type": "mule_ring",
        })
        txn_counter += 1
    return pd.DataFrame(rows), txn_counter, mule_wallets


def inject_account_takeover(users_df, social_graph, txn_counter, n_cases=60):
    """New device + new location + rapid-fire transfers draining the
    balance to recipients the victim has never sent to before."""
    rows = []
    victims = users_df.sample(n_cases, random_state=2)
    for _, v in victims.iterrows():
        day = random.randint(3, N_DAYS - 1)
        base_ts = SIM_START + timedelta(days=day, hours=random.randint(0, 23))
        new_device = f"D{uuid.uuid4().hex[:8]}"  # attacker's device, not the victim's
        n_burst = random.randint(3, 6)
        for k in range(n_burst):
            ts = base_ts + timedelta(minutes=k * random.randint(1, 4))
            new_recipient = f"M{uuid.uuid4().hex[:6]}"
            amount = float(np.round(v["typical_amount"] * random.uniform(2, 6), 2))
            rows.append({
                "txn_id": f"T{txn_counter:07d}", "timestamp": ts.isoformat(),
                "sender_id": v["user_id"], "sender_type": "user",
                "receiver_id": new_recipient, "receiver_type": "user",
                "channel": "send_money", "amount": amount,
                "sender_district": random.choice([d for d in DISTRICTS if d != v["district"]]),
                "device_id": new_device,
                "is_scam": 1, "scam_type": "account_takeover",
            })
            txn_counter += 1
    return pd.DataFrame(rows), txn_counter


def main():
    print("Generating users and agents...")
    users_df = make_users(N_USERS)
    agents_df = make_agents(N_AGENTS)
    social_graph = build_social_graph(users_df)

    print("Generating normal transaction history...")
    normal_df, txn_counter = gen_normal_transactions(users_df, agents_df, social_graph, N_DAYS)

    print("Injecting social-engineering scam cases...")
    se_df, txn_counter, se_wallets = inject_social_engineering(users_df, txn_counter)

    print("Injecting mule-ring cases...")
    mule_df, txn_counter, mule_wallets = inject_mule_rings(users_df, txn_counter)

    print("Injecting account-takeover cases...")
    ato_df, txn_counter = inject_account_takeover(users_df, social_graph, txn_counter)

    all_txns = pd.concat([normal_df, se_df, mule_df, ato_df], ignore_index=True)
    all_txns["timestamp"] = pd.to_datetime(all_txns["timestamp"])
    all_txns = all_txns.sort_values("timestamp").reset_index(drop=True)

    users_df.to_csv("users.csv", index=False)
    agents_df.to_csv("agents.csv", index=False)
    all_txns.to_csv("transactions.csv", index=False)

    with open("known_bad_wallets.txt", "w") as f:
        for w in se_wallets + mule_wallets:
            f.write(w + "\n")

    print(f"\nDone.")
    print(f"  users.csv         : {len(users_df)} rows")
    print(f"  agents.csv        : {len(agents_df)} rows")
    print(f"  transactions.csv  : {len(all_txns)} rows "
          f"({all_txns['is_scam'].sum()} labeled suspicious, "
          f"{all_txns['is_scam'].mean()*100:.2f}%)")
    print(f"  scam_type breakdown:\n{all_txns[all_txns.is_scam==1]['scam_type'].value_counts()}")


if __name__ == "__main__":
    main()
