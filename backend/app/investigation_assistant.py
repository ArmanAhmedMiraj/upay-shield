"""
upay Shield - AI Investigation Assistant
--------------------------------------------
Takes the STRUCTURED evidence already computed by the risk model and the
graph analysis (never raw free text, never unverified claims) and writes a
short case summary an analyst can read in a few seconds.

Two modes:
  1. template_explain()   - always available, zero dependencies, deterministic.
                             Used as the default / offline fallback.
  2. llm_explain()        - calls the Claude API to turn the same structured
                             evidence into a more natural narrative. Only used
                             if ANTHROPIC_API_KEY is set in the environment.
                             The model is ONLY given the structured evidence
                             below -- never raw account data -- so it cannot
                             invent facts that aren't in the evidence.

This mirrors the rulebook's "AI investigation assistant" direction:
"explain why an alert was generated and summarize the evidence for an
analyst," and follows the Responsible AI requirement to keep predictions,
assumptions, and generated text clearly separated (the evidence dict is the
source of truth; the text is just a readable wrapper around it).
"""

import os
import json

SCAM_TYPE_LABELS = {
    "social_engineering": "a scam-call style transfer (large, one-off payment to a brand-new recipient)",
    "mule_ring": "a money-mule pass-through pattern",
    "account_takeover": "a possible account takeover (new device/location, rapid transfers)",
    "unknown": "an unusual transaction",
}


def _risk_level(score: float) -> str:
    if score >= 0.7:
        return "HIGH"
    if score >= 0.4:
        return "MEDIUM"
    return "LOW"


def _guess_scam_type(evidence: dict) -> str:
    f = evidence.get("features", {})
    if f.get("receiver_indegree_1h", 0) >= 3 or evidence.get("mule_evidence"):
        return "mule_ring"
    if f.get("new_device_for_sender") and f.get("sender_velocity_10min", 0) >= 2:
        return "account_takeover"
    if f.get("is_new_recipient") and f.get("amount_vs_avg_ratio", 1) >= 3:
        return "social_engineering"
    return "unknown"


def template_explain(txn: dict, risk_score: float, feature_contributions: list,
                      mule_evidence: dict | None) -> dict:
    """Deterministic, rule-based explanation. No external calls, no
    hallucination risk -- every sentence is traceable to a specific field."""
    evidence = {"features": txn.get("features", {}), "mule_evidence": mule_evidence}
    scam_type = _guess_scam_type({"features": txn.get("features", {}), "mule_evidence": mule_evidence})
    level = _risk_level(risk_score)

    lines = []
    lines.append(
        f"Transaction {txn.get('txn_id', '')} flagged as {level} RISK "
        f"(score {risk_score:.0%}), most consistent with {SCAM_TYPE_LABELS.get(scam_type, scam_type)}."
    )

    feats = txn.get("features", {})
    detail_bits = []
    if feats.get("is_new_recipient"):
        detail_bits.append("the sender has never sent money to this recipient before")
    if feats.get("amount_vs_avg_ratio", 1) >= 3:
        detail_bits.append(
            f"the amount is {feats['amount_vs_avg_ratio']:.1f}x the sender's usual transfer size"
        )
    if feats.get("unusual_hour"):
        detail_bits.append("it happened late at night (11pm-5am), outside typical hours")
    if feats.get("new_device_for_sender"):
        detail_bits.append("it came from a device not previously associated with this sender")
    if feats.get("sender_velocity_10min", 0) >= 2:
        detail_bits.append(
            f"the sender made {int(feats['sender_velocity_10min'])} other transactions in the "
            f"preceding 10 minutes (rapid-fire pattern)"
        )
    if feats.get("receiver_age_days") is not None and feats["receiver_age_days"] <= 1:
        detail_bits.append("the receiving wallet has no transaction history before this")

    if detail_bits:
        lines.append("Evidence: " + "; ".join(detail_bits) + ".")

    if mule_evidence:
        lines.append(
            "Network evidence: the receiving wallet also matches a mule-network pattern — "
            + "; ".join(mule_evidence.get("reasons", [])) + "."
        )

    recommendation = {
        "HIGH": "Recommend holding this transaction for manual review before release.",
        "MEDIUM": "Recommend a step-up confirmation from the customer before release.",
        "LOW": "No action needed; informational only.",
    }[level]
    lines.append(recommendation)

    return {
        "risk_level": level,
        "scam_type_guess": scam_type,
        "summary": " ".join(lines),
        "recommendation": recommendation,
        "evidence_used": evidence,
    }


LLM_SYSTEM_PROMPT = """You are a fraud-investigation assistant for a mobile \
financial service. You will be given STRUCTURED EVIDENCE (JSON) about one \
flagged transaction: model risk score, feature values, and graph/network \
evidence if any. Write a short (3-5 sentence) case note for a human fraud \
analyst, in clear plain English suitable for someone who is not a data \
scientist.

Rules:
- Use ONLY the facts given in the evidence JSON. Do not invent names, \
amounts, locations, or claims not present in the evidence.
- State the risk level and the single most likely scam pattern.
- List the 2-4 strongest pieces of evidence in plain language.
- End with one clear recommended action for the analyst (hold for review, \
request step-up confirmation, or no action).
- Do not use the word "I" or add disclaimers. Write it as a case note, not \
a conversation."""


def llm_explain(txn: dict, risk_score: float, mule_evidence: dict | None) -> dict:
    """Optional: calls the Claude API for a more natural narrative, grounded
    strictly in the same structured evidence used by template_explain().
    Falls back to the template if no API key is configured or the call fails.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    template_result = template_explain(txn, risk_score, [], mule_evidence)
    if not api_key:
        return template_result

    try:
        import anthropic  # only imported if a key is actually configured
        client = anthropic.Anthropic(api_key=api_key)
        evidence_payload = {
            "txn_id": txn.get("txn_id"),
            "risk_score": round(risk_score, 3),
            "features": txn.get("features", {}),
            "mule_network_evidence": mule_evidence,
        }
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(evidence_payload, indent=2)}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        template_result["summary"] = text.strip()
        template_result["generated_by"] = "claude-sonnet-4-6"
        return template_result
    except Exception as e:
        template_result["llm_error"] = str(e)
        return template_result


if __name__ == "__main__":
    # quick smoke test with a fabricated example
    demo_txn = {
        "txn_id": "T0099999",
        "features": {
            "is_new_recipient": 1,
            "amount_vs_avg_ratio": 8.2,
            "unusual_hour": 1,
            "new_device_for_sender": 0,
            "sender_velocity_10min": 0,
            "receiver_age_days": 0.1,
            "receiver_indegree_1h": 9,
        },
    }
    demo_mule_evidence = {
        "wallet": "Mb4170b",
        "mule_score": 0.85,
        "reasons": [
            "11 distinct senders within a single hour (very high burst)",
            "relatively new wallet (29 days of history)",
            "97% of received funds moved back out within 26 minutes of the last incoming transfer",
        ],
    }
    result = template_explain(demo_txn, 0.93, [], demo_mule_evidence)
    print(json.dumps(result, indent=2))
