# CLAUDE.md — upay Shield

AI DEV FEST 2026 (DIU-CPC × upay), Track 01: Trust & Risk Intelligence.
This file is what Claude Code reads automatically in this repo. It covers
both the hackathon's rules and this specific codebase.

## What this project is

Fraud/scam/mule-network detection for an MFS: a risk-scoring model, a
transaction-graph mule detector, an AI investigation assistant, and a
Flask + HTML/JS analyst dashboard. Full details: `README.md` and
`PROJECT_REPORT.md`.

## Codebase map

```
data/generate_data.py            synthetic data generator (run first)
backend/app/features.py          feature engineering
backend/app/train_model.py       trains the risk model
backend/app/graph_analysis.py    mule-network detection
backend/app/investigation_assistant.py   case-summary generation
backend/app/api.py               Flask API + serves the dashboard
backend/app/generate_demo_snapshot.py    precomputes frontend/demo_data.js
frontend/dashboard.html          the analyst dashboard (single file)
```

Pipeline order matters: `generate_data.py` → `features.py` →
`train_model.py` → `graph_analysis.py` → `generate_demo_snapshot.py` →
then `api.py` to serve it. Re-run this chain after changing the generator,
features, or model.

## Rule 1 — Commit step by step (rulebook §5)

The repo is PUBLIC on GitHub. Judges check for a continuous commit
history. One feature/fix/improvement per commit, clear messages (e.g.
`feat: add agent risk scoring`, `fix: handle missing device_id`). Never
batch a day of work into one commit — push after each real step,
including during the on-site final's new-requirements phase.

## Rule 2 — No secrets in the repo (§6)

Keys go only in `.env` (gitignored). Keep `.env.example` current with
every variable name and a placeholder.

## Rule 3 — README.md stays complete (§6)

Update `README.md` whenever a feature, dependency, env var, or command
changes. It must always cover: overview, features + how AI is used, tech
stack, requirements, install/setup, env vars, run/build commands, live
deployment URL, testing instructions, other configuration.

## Rule 4 — Explain everything (§4.5)

Prefer readable code over clever code. When adding or changing an AI
component, explain briefly in chat how it works and why. Known-honest
limitations belong in the README, not hidden.

## Rule 5 — Responsible AI (§14)

- Synthetic/self-generated data only — never add real upay or customer data.
- Every risk score must stay traceable to specific feature values (no
  opaque black-box outputs).
- Agents are excluded from mule-network scoring (see `graph_analysis.py`)
  — don't remove that exclusion without re-checking the false-positive rate.
- The model/API must never auto-block a transaction. High risk = held for
  human review, always. Don't change this without flagging it explicitly.
- Keep predictions, assumptions, and AI-generated text in separate fields
  in API responses — never merge them into one opaque blob.

## Rule 6 — Build for change (§8)

Keep AI logic, business rules (risk thresholds/actions), API routes, and
the dashboard UI in separate files/layers — the on-site final adds new
requirements, so modularity matters more than cleverness.

## Submission checklist (§7) — by T+72h

- [ ] Public GitHub repo, all code pushed, continuous commit history
- [ ] `README.md` complete per Rule 3
- [ ] `PROJECT_REPORT.md` complete
- [ ] Live deployment URL added to `README.md`
- [ ] Video demo recorded: how it works, AI components, real-life impact
- [ ] Any extra materials the organizers request
