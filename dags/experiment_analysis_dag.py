"""
Experiment Analysis DAG - Daily experiment analysis.

Reads newest experiment run in data/experiments/, runs analyze.py + report.py,
saves artifacts to artifacts/experiments/ and logs key outputs.
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Project root - adjust if DAG runs from different location
PROJECT_ROOT = Path(__file__).parent.parent


def _find_latest_experiment(**kwargs):
    """Find most recently modified experiment directory."""
    exp_dir = PROJECT_ROOT / "data" / "experiments"
    if not exp_dir.exists():
        return None
    dirs = [d for d in exp_dir.iterdir() if d.is_dir() and (d / "participants.csv").exists()]
    if not dirs:
        dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
    if not dirs:
        return None
    latest = max(dirs, key=lambda p: p.stat().st_mtime)
    return latest.name


def _run_analysis_and_report(exp_id: str, **kwargs):
    """Run analyze and report for given experiment."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    
    from src.experimentation.analyze import run_analysis
    from src.experimentation.report import render_exec_summary
    
    data_dir = str(PROJECT_ROOT / "data" / "experiments")
    artifacts_dir = str(PROJECT_ROOT / "artifacts" / "experiments")
    
    result = run_analysis(
        experiment_id=exp_id,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        segment_cols=["tenure", "Contract", "InternetService"],
    )
    
    render_exec_summary(result.to_dict(), exp_id, artifacts_dir=artifacts_dir)
    return {"experiment_id": exp_id, "recommendation": result.recommendation}


default_args = {
    "owner": "data-science",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "experiment_analysis_daily",
    default_args=default_args,
    description="Daily experiment analysis: analyze + report",
    schedule="0 6 * * *",  # 6 AM daily
    max_active_runs=1,
    tags=["experiment", "analysis", "retention"],
)

find_task = PythonOperator(
    task_id="find_latest_experiment",
    python_callable=_find_latest_experiment,
    dag=dag,
)


def _run_analysis_wrapper(**kwargs):
    ti = kwargs.get("ti")
    exp_id = ti.xcom_pull(task_ids="find_latest_experiment")
    if not exp_id:
        return {"status": "skipped", "reason": "no experiment found"}
    return _run_analysis_and_report(exp_id=exp_id, **kwargs)


analyze_task = PythonOperator(
    task_id="run_analysis_report",
    python_callable=_run_analysis_wrapper,
    dag=dag,
)

# Only run analyze if we found an experiment
# Simplified: always run find, then analyze (analyze will no-op if no exp)
find_task >> analyze_task
