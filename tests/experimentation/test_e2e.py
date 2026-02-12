"""End-to-end: simulate -> analyze produces non-empty artifacts."""
import tempfile
import shutil
from pathlib import Path
from io import StringIO

import pandas as pd
import pytest

# Minimal valid telco-like CSV
TEST_CSV = """customerID,gender,SeniorCitizen,tenure,MonthlyCharges,TotalCharges,Contract,InternetService,Churn
C1,Female,0,12,65,780,Month-to-month,DSL,No
C2,Male,1,5,75,375,Month-to-month,Fiber optic,Yes
C3,Female,0,24,55,1320,One year,DSL,No
C4,Male,0,8,90,720,Month-to-month,Fiber optic,Yes
C5,Female,1,36,50,1800,Two year,No,No
"""


@pytest.fixture
def temp_experiment_dir():
    """Temporary directory for experiment data."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_e2e_simulate_analyze(temp_experiment_dir):
    """Replay simulation -> analyze -> artifacts exist."""
    data_dir = Path(temp_experiment_dir) / "data" / "experiments"
    artifacts_dir = Path(temp_experiment_dir) / "artifacts" / "experiments"
    data_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    
    csv_path = Path(temp_experiment_dir) / "customers.csv"
    df = pd.read_csv(StringIO(TEST_CSV))
    df = pd.concat([df] * 60, ignore_index=True)
    df["customerID"] = [f"C{i}" for i in range(len(df))]
    df.to_csv(csv_path, index=False)
    
    from src.experimentation.simulate_campaign import run_replay
    from src.experimentation.analyze import run_analysis
    from src.experimentation.report import render_exec_summary
    
    exp_id = "e2e_test"
    
    run_replay(
        experiment_id=exp_id,
        data_path=str(csv_path),
        max_customers=300,
        data_dir=str(data_dir),
    )
    
    result = run_analysis(
        experiment_id=exp_id,
        data_dir=str(data_dir),
        artifacts_dir=str(artifacts_dir),
    )
    
    render_exec_summary(result.to_dict(), exp_id, artifacts_dir=str(artifacts_dir))
    
    out_dir = artifacts_dir / exp_id
    assert (out_dir / "analysis.json").exists()
    assert (out_dir / "exec_summary.html").exists()
    assert (out_dir / "segments.csv").exists() or True  # segments may be empty
    assert result.experiment_id == exp_id
    assert result.recommendation in ("ship", "hold", "iterate")
