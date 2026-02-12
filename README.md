# Telco Churn MLOps + A/B Experimentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Production-grade telco churn prediction pipeline with BCG-style experimentation, A/B testing, and retention decision support.

---

## Overview

End-to-end **client-facing decision product** for telco retention:

| Feature | Description |
|--------|-------------|
| **Churn Pipeline** | sklearn GradientBoosting, MLflow, Flask API, Kafka-ready |
| **Experimentation** | SRM checks, power/MDE, CUPED, z-test/t-test, segment analysis |
| **Simulator** | Replay + model-based retention campaign (no proprietary logs) |
| **Dashboard** | Streamlit UI: Design, Run, Results, Recommendation |
| **Reports** | One-page HTML executive summary (Jinja2) |
| **Orchestration** | Airflow DAG for daily experiment analysis |

---

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/telco-churn-experimentation.git
cd telco-churn-experimentation

# Setup
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
pip install -e .


# Run demo (no model training needed)
make experiment-demo

# Launch dashboard
make experiment-dashboard
# Open http://localhost:8501
```

---

## Project Structure

```
├── src/
│   ├── experimentation/       # A/B testing module
│   │   ├── schema.py          # ExperimentConfig, Assignment, Events
│   │   ├── assignment.py      # Deterministic hash-based assignment
│   │   ├── event_store.py     # Exposure/outcome storage
│   │   ├── analyze.py         # SRM, lift, CI, CUPED, segments
│   │   ├── report.py          # Executive summary HTML
│   │   ├── simulate_campaign.py
│   │   └── stats/             # srm, power, cuped, hypothesis_tests, segments
│   ├── api/                   # Flask inference API
│   └── ...
├── apps/
│   └── experiment_dashboard.py
├── dags/
│   └── experiment_analysis_dag.py
├── scripts/
│   └── run_experiment_demo.py
├── tests/
│   └── experimentation/
├── data/raw/                  # Telco-Customer-Churn.csv
├── artifacts/experiments/     # analysis.json, exec_summary.html, plots
├── requirements.txt
├── Makefile
├── LICENSE                    # MIT
└── README.md
```

---

## Commands

| Command | Description |
|---------|-------------|
| `make train` | Train churn model |
| `make experiment-demo` | Simulate + analyze + report |
| `make experiment-dashboard` | Start Streamlit on port 8501 |
| `make experiment-analyze EXP_ID=xxx` | Analyze experiment |
| `make test` | Run pytest |
| `make install` | Install dependencies |

---

## Artifacts

After `make experiment-demo`:

```
artifacts/experiments/demo_replay_001/
├── analysis.json       # Full analysis (SRM, lift, p-value, CI)
├── exec_summary.html   # One-page executive memo
├── lift_chart.png      # Control vs Treatment
├── lift_ci.png         # Lift with 95% CI
└── segments.csv        # Per-segment breakdown
```

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| ML | scikit-learn, pandas, numpy, scipy |
| Experimentation | SRM, CUPED, z-test, t-test, power analysis |
| Dashboard | Streamlit |
| API | Flask, Waitress |
| Orchestration | Apache Airflow |
| Storage | Parquet / CSV, Kafka-ready |

---

## Testing

```bash
pytest tests/experimentation -v
```

---

## License

MIT License. See [LICENSE](LICENSE).

---
