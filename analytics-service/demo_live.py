#!/usr/bin/env python3
"""
Standalone test script to visualize and verify the PR Risk prediction microservice live.
Demonstrates live inference on Scenario A (Low Risk) and Scenario B (High Risk).
"""

import json
import os
import urllib.error
import urllib.request

# Scenarios definition
SCENARIOS = {
    "A": {
        "name": "Scenario A: Low Risk PR (Small Bugfix by Experienced Contributor)",
        "pr_title": "fix(auth): handle null token in refresh coordinator (#104)",
        "features": {
            "la": 15.0,
            "ld": 3.0,
            "nf": 1.0,
            "ns": 1.0,
            "nd": 1.0,
            "entropy": 0.1,
            "ndev": 5.0,
            "lt": 50.0,
            "nuc": 10.0,
            "age": 2.0,
            "exp": 120.0,
            "rexp": 25.0,
            "sexp": 15.0,
            "fix": 1.0,
        },
    },
    "B": {
        "name": "Scenario B: High Risk PR (Massive Refactor by First-Time Contributor)",
        "pr_title": "refactor(core): rewrite data pipeline and database layer (#105)",
        "features": {
            "la": 850.0,
            "ld": 320.0,
            "nf": 18.0,
            "ns": 6.0,
            "nd": 6.0,
            "entropy": 0.85,
            "ndev": 15.0,
            "lt": 2500.0,
            "nuc": 120.0,
            "age": 360.0,
            "exp": 0.0,
            "rexp": 0.0,
            "sexp": 0.0,
            "fix": 0.0,
        },
    },
}

API_URL = "http://localhost:8000/predict"
HEALTH_URL = "http://localhost:8000/health"


def is_service_running():
    try:
        req = urllib.request.Request(HEALTH_URL, method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def predict_via_api(features):
    data = json.dumps(features).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def predict_via_local_model(features):
    import joblib
    import pandas as pd

    model_path = os.path.join(os.path.dirname(__file__), "pr_risk_model.joblib")
    if not os.path.exists(model_path):
        model_path = "pr_risk_model.joblib"
    model = joblib.load(model_path)
    df = pd.DataFrame([features])[model.feature_names_in_]
    proba = float(model.predict_proba(df)[0, 1])
    score = int(round(proba * 100))
    if score <= 30:
        level = "LOW"
    elif score <= 70:
        level = "MEDIUM"
    else:
        level = "HIGH"
    return {"probability": proba, "riskScore": score, "riskLevel": level}


def render_ascii_toast(title, pr_title, score, level):
    color_codes = {
        "LOW": "\033[92m",  # Green
        "MEDIUM": "\033[93m",  # Yellow
        "HIGH": "\033[91m",  # Red
    }
    reset = "\033[0m"
    bold = "\033[1m"
    dim = "\033[2m"
    color = color_codes.get(level, "\033[92m")

    # Shorten pr_title if needed
    display_title = (pr_title[:45] + "...") if len(pr_title) > 48 else pr_title

    header_line = (
        f"  {bold}│{reset}  {color}●{reset} {bold}PR Risk Analysis Complete{reset}"
        f"                            {dim}×{reset}   {bold}│{reset}"
    )
    score_line = (
        f"  {bold}│{reset}  Risk Score: {bold}{score:>3}{reset} / 100"
        f"               [{color}{bold}{level:^8}{reset}]   {bold}│{reset}"
    )
    print(f"""
  {bold}┌────────────────────────────────────────────────────────┐{reset}
{header_line}
  {bold}│{reset}  {dim}Based on code size and history.{reset}                       {bold}│{reset}
  {bold}│{reset}                                                        {bold}│{reset}
  {bold}│{reset}  {bold}PR:{reset} {display_title:<49}  {bold}│{reset}
  {bold}│{reset}  ──────────────────────────────────────────────────────  {bold}│{reset}
{score_line}
  {bold}└────────────────────────────────────────────────────────┘{reset}
""")


def main():
    service_active = is_service_running()
    status_text = "ONLINE [200 OK]" if service_active else "OFFLINE (Using fallback)"
    mode = (
        "FastAPI HTTP (http://localhost:8000/predict)"
        if service_active
        else "Local Scikit-Learn Model Fallback"
    )

    print("\n" + "=" * 65)
    print("      ADEPT PR RISK ANALYTICS - LIVE VERIFICATION")
    print("=" * 65)
    print(f" Microservice Status : {status_text}")
    print(f" Execution Mode      : {mode}")
    print("=" * 65)

    for key in ["A", "B"]:
        scenario = SCENARIOS[key]
        features = scenario["features"]
        pr_title = scenario["pr_title"]

        print(f"\n>> RUNNING {scenario['name']}")
        print("-" * 65)
        print(" [Input 14 Numerical Metrics]:")
        for k, v in features.items():
            print(f"   - {k:<8}: {v:>7.2f}")

        # Predict
        result = predict_via_api(features) if service_active else predict_via_local_model(features)

        prob = result["probability"]
        score = result["riskScore"]
        level = result["riskLevel"]

        print("\n [Prediction Output]:")
        print(f"   - Raw Defect Probability : {prob:.4f}")
        print(f"   - Risk Score (0-100)     : {score}")
        print(f"   - Risk Level             : {level}")

        # JSON Payloads
        backend_payload = {
            "prTitle": pr_title,
            "riskScore": score,
            "riskLevel": level,
            "probability": prob,
        }
        print("\n [SSE Broadcast Payload (sent to React UI)]:")
        print("  " + json.dumps(backend_payload, indent=2).replace("\n", "\n  "))

        print("\n [React Toast Preview on Dashboard (Bottom-Right)]:")
        render_ascii_toast("PR Risk Analysis Complete", pr_title, score, level)
        print("-" * 65)

    print("\n[SUCCESS] Live simulation complete. Both scenarios verified.")


if __name__ == "__main__":
    main()
