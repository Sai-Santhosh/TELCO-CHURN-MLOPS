#!/usr/bin/env python3
"""
Run full experiment demo: simulate -> analyze -> report.

Creates artifacts/experiments/<id>/analysis.json, exec_summary.html, plots, segment table.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

def main():
    experiment_id = "demo_replay_001"
    data_dir = ROOT / "data" / "experiments"
    artifacts_dir = ROOT / "artifacts" / "experiments"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Raw data path - try multiple locations
    raw_paths = [
        ROOT / "data" / "raw" / "Telco-Customer-Churn.csv",
        Path("data/raw/Telco-Customer-Churn.csv"),
        ROOT.parent / "data" / "raw" / "Telco-Customer-Churn.csv",
        Path(__file__).parent.parent.parent / "data" / "raw" / "Telco-Customer-Churn.csv",  # repo root
    ]
    raw_path = None
    for p in raw_paths:
        if Path(p).exists():
            raw_path = str(p)
            break
    
    if not raw_path:
        print("ERROR: Telco-Customer-Churn.csv not found. Place it in data/raw/")
        sys.exit(1)
    
    print("1. Running replay simulation...")
    from src.experimentation.simulate_campaign import run_replay
    summary = run_replay(
        experiment_id=experiment_id,
        data_path=raw_path,
        max_customers=1500,
        data_dir=str(data_dir),
    )
    print(f"   Assigned: {summary['n_control']} control, {summary['n_treatment']} treatment")
    
    print("2. Running analysis...")
    from src.experimentation.analyze import run_analysis
    result = run_analysis(
        experiment_id=experiment_id,
        data_dir=str(data_dir),
        artifacts_dir=str(artifacts_dir),
        segment_cols=["tenure", "Contract", "InternetService"],
    )
    
    print("3. Generating executive summary...")
    from src.experimentation.report import render_exec_summary
    from src.experimentation.event_store import read_participants
    
    out_path = render_exec_summary(
        result.to_dict(),
        experiment_id,
        artifacts_dir=str(artifacts_dir),
    )
    
    out_dir = artifacts_dir / experiment_id
    print(f"\n[OK] Demo complete. Artifacts in {out_dir}:")
    for f in sorted(out_dir.iterdir()):
        print(f"   - {f.name}")

if __name__ == "__main__":
    main()
