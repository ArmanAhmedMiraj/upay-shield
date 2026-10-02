# upay Shield — Project Report

**Track 01: Trust & Risk Intelligence · AI DEV FEST 2026 · DIU CPC × upay**

## 1. The problem

Scam money in a modern MFS usually wins the race against detection. A
victim is socially engineered (impersonation calls, fake prizes,
"wrong-transfer, please return it" tricks) into sending money themselves —
no OTP or PIN theft required. The funds then move through one or more
wallets and are cashed out at an agent, often within the hour. By the time
a victim reports it, there is nothing left to recover and no clear trail
for an analyst to follow quickly.

**Who experiences this:** customers who lose money directly; the upay
trust & risk team, who investigate each case manually after the fact with
no automated way to see the fund-movement pattern; and the wider MFS
ecosystem, whose growth depends on customers trusting that their money is
safe (the rulebook's own framing: "trust is foundational to wallet adoption
and transaction growth").

## 2. The proposed idea

upay Shield is a three-layer system:

1. **Real-time risk scoring** — every transaction gets a 0–100% risk score
   the instant it happens, based on how it compares to the sender's own
   normal pattern and the receiving wallet's behavior.
2. **Mule network discovery** — a transaction graph identifies wallets that
   show the classic money-mule shape: money arriving fast from many
   unrelated people, then leaving again almost as fast.
3. **AI investigation assistant** — every flagged case gets an automatically
   generated, evidence-grounded summary so a human analyst can decide in
   seconds instead of minutes, and a clear hold/approve/dismiss action.

The key design decision: **the AI never blocks money by itself.** It holds
high-risk transactions for a human to review, exactly as the rulebook's
"no harmful automation" principle requires.

## 3. Implemented solution

- `data/generate_data.py` — synthetic MFS data generator: 2,000 users, 120
  agents, ~97,000 transactions over 60 days, with three labeled scam
  patterns deliberately injected (social engineering, mule rings, account
  takeover) so the model has real signal to learn from.
- `backend/app/features.py` — turns raw transactions into explainable
  behavioral and network features (new recipient? unusual amount for this
  sender? new device? how old is the receiving wallet? how many strangers
  paid it in the last hour?).
- `backend/app/train_model.py` — trains a gradient-boosted classifier and
  evaluates it on a held-out test set that was never used for training.
- `backend/app/graph_analysis.py` — builds the transaction graph and scores
  wallets for mule-like behavior, with agents deliberately excluded (their
  legitimately high turnover would otherwise look identical to a mule).
- `backend/app/investigation_assistant.py` — generates the case summary
  from structured evidence only (template-based by default; optional
  Claude API mode for a more natural narrative, same evidence either way).
- `backend/app/api.py` + `frontend/dashboard.html` — the working prototype:
  a Flask API and a single-page analyst dashboard.

## 4. Key features

- Live risk scoring with a transparent feature breakdown per transaction
- Mule-cluster visualization (who is feeding money into a flagged wallet)
- Plain-language, evidence-grounded case summaries for analysts
- Approve / hold / dismiss workflow — human-in-the-loop by design
- Works fully offline too: the dashboard ships with a precomputed snapshot
  of real results, so it's never empty even without the backend running

## 5. AI approach

| Task | Method | Why |
|---|---|---|
| Transaction risk classification | Gradient-boosted decision trees | Matches the rulebook's own suggested approach for this problem; fast, and every decision can be traced back to specific feature values |
| Mule-network discovery | Graph analytics (fan-in burst, pass-through timing, connected components) | A human-interpretable approach rather than a black-box graph neural net, so an analyst can see exactly why a wallet was flagged |
| Case narrative generation | Rule-based template grounded in structured evidence, with an optional Claude API mode | Never invents facts not present in the evidence — the evidence dict is the single source of truth either way |

A genuine finding from development: the first version of the risk model
scored a suspicious 99.97% AUC. Digging in showed it had learned an overly
narrow rule on wallet age (0.0 vs 0.2 days flipped the prediction
completely). This was fixed by bucketing the feature into robust bands and
adding more training diversity — documented in full in the README's "Known
limitations" section, because a too-good number on synthetic data is a
reason to investigate, not a reason to stop.

## 6. Intended real-life impact

- **For customers:** transactions that match a known scam pattern get held
  before the money disappears, instead of being investigated after it's
  gone.
- **For upay's trust & risk team:** investigation time per case drops from
  reading through raw transaction logs to reading a 3–5 sentence case
  summary with the evidence already laid out.
- **For the ecosystem:** a visible, explainable fraud-prevention layer is a
  direct contributor to the trust that the rulebook identifies as
  foundational to wallet adoption and transaction growth.

**Measurable success metrics** (see `data/evaluation.json` for current
numbers on synthetic data): precision and recall of the risk model against
known scam cases; number of mule wallets correctly identified before
cash-out; reduction in average time-to-decision for flagged transactions.

## 7. Path to validation (post-hackathon)

See the README's "Path to validation & scale" section for the full detail:
controlled validation against anonymized real transaction samples,
threshold tuning against real investigation capacity, multi-hop mule
tracing, and a production-grade decision store with audit logging.
