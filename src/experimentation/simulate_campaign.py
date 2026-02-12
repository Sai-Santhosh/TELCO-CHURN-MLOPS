"""
Retention Campaign Simulator.

Uses churn model probabilities to identify high-risk customers, assigns
control/treatment via assignment module, and simulates outcomes:
- baseline churn prob = model predicted prob
- treatment reduces churn by configurable relative effect (e.g., 10-20%)
- generates binary churn outcome + optional continuous proxy (revenue saved)

Writes ExposureEvent and OutcomeEvent to event_store. Returns run summary.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .schema import ArmType, ExperimentConfig, ExposureEvent, OutcomeEvent
from .assignment import assign_entities
from .event_store import append_exposures, append_outcomes, write_participants

logger = logging.getLogger(__name__)

SIMULATOR_SEED = 42


def load_churn_model(model_path: str):
    """Load trained churn pipeline."""
    import joblib
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    obj = joblib.load(path)
    return obj


def predict_churn_probs(model, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Get churn probabilities from model. Handles various model formats."""
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        available = [c for c in df.columns if c not in ("customerID", "Churn", "ChurnLabel")]
    X = df[available]
    
    pipeline = model
    if isinstance(model, dict):
        pipeline = model.get("pipeline", model.get("model", model))
    
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raise ValueError("Model must support predict_proba")


def run_retention_simulation(
    experiment_id: str,
    customers_df: pd.DataFrame,
    churn_probs: np.ndarray,
    config: Optional[ExperimentConfig] = None,
    treatment_effect_rel: float = 0.15,
    noise_std: float = 0.05,
    entity_id_col: str = "customerID",
    segment_cols: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    data_dir: str = "data/experiments",
    random_seed: int = SIMULATOR_SEED,
) -> Dict:
    """
    Run retention campaign simulation.
    
    Args:
        experiment_id: Experiment identifier
        customers_df: DataFrame with customer attributes
        churn_probs: Array of baseline churn probabilities (from model)
        config: ExperimentConfig (optional)
        treatment_effect_rel: Relative churn reduction for treatment (e.g. 0.15 = 15%)
        noise_std: Noise added to treatment effect
        entity_id_col: Column with customer ID
        segment_cols: Columns for stratification (tenure, Contract, etc.)
        model_path: Path to model (for covariate; optional)
        data_dir: Base directory for experiment data
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict with n_assigned, n_control, n_treatment, exposures_written, outcomes_written
    """
    np.random.seed(random_seed)
    
    if config is None:
        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=f"Retention campaign {experiment_id}",
            allocation=0.5,
        )
    
    entity_ids = customers_df[entity_id_col].astype(str).tolist()
    
    stratify_by = None
    if segment_cols:
        seg_col = segment_cols[0] if segment_cols else None
        if seg_col and seg_col in customers_df.columns:
            stratify_by = dict(zip(entity_ids, customers_df[seg_col].astype(str)))
    
    assignments = assign_entities(entity_ids, config, stratify_by)
    
    now = datetime.utcnow()
    
    exposures = []
    outcomes = []
    
    churn_probs = np.asarray(churn_probs).flatten()
    if len(churn_probs) != len(customers_df):
        raise ValueError("churn_probs length must match customers_df")
    
    entity_to_row = dict(zip(customers_df[entity_id_col].astype(str), range(len(customers_df))))
    assign_by_entity = {a.entity_id: a for a in assignments}
    
    for eid, row_idx in entity_to_row.items():
        a = assign_by_entity.get(eid)
        if not a:
            continue
        
        base_prob = np.clip(churn_probs[row_idx], 0.01, 0.99)
        
        if a.arm == ArmType.TREATMENT:
            effect = 1 - treatment_effect_rel + np.random.normal(0, noise_std)
            effect = np.clip(effect, 0.5, 1.0)
            actual_prob = base_prob * effect
        else:
            actual_prob = base_prob
        
        actual_prob = np.clip(actual_prob, 0, 1)
        churned = 1 if np.random.random() < actual_prob else 0
        
        exposures.append(ExposureEvent(
            experiment_id=experiment_id,
            entity_id=eid,
            arm=a.arm,
            exposed_at=now,
            metadata={"base_churn_prob": float(base_prob)},
        ))
        
        outcomes.append(OutcomeEvent(
            experiment_id=experiment_id,
            entity_id=eid,
            arm=a.arm,
            outcome_name="churned",
            outcome_value=float(churned),
            observed_at=now,
            metadata={"churn_prob": float(actual_prob)},
        ))
    
    n_exp = append_exposures(exposures, experiment_id, base_dir=data_dir)
    n_out = append_outcomes(outcomes, experiment_id, base_dir=data_dir)
    
    seg_cols = segment_cols or []
    parts_data = []
    for e, o in zip(exposures, outcomes):
        row = {"entity_id": e.entity_id, "arm": e.arm.value, "outcome_value": o.outcome_value}
        entity_row_idx = entity_to_row.get(e.entity_id)
        if entity_row_idx is not None:
            cust_row = customers_df.iloc[entity_row_idx]
            for c in seg_cols:
                if c in cust_row.index:
                    row[c] = cust_row[c]
        parts_data.append(row)
    parts_df = pd.DataFrame(parts_data)
    write_participants(parts_df, experiment_id, base_dir=data_dir)
    
    n_control = sum(1 for a in assignments if a.arm == ArmType.CONTROL)
    n_treatment = sum(1 for a in assignments if a.arm == ArmType.TREATMENT)
    
    summary = {
        "experiment_id": experiment_id,
        "n_assigned": len(assignments),
        "n_control": n_control,
        "n_treatment": n_treatment,
        "exposures_written": n_exp,
        "outcomes_written": n_out,
        "treatment_effect_rel": treatment_effect_rel,
        "random_seed": random_seed,
    }
    
    logger.info(f"Simulation complete: {summary}")
    return summary


def run_from_model(
    experiment_id: str,
    data_path: str = "data/processed/sample.csv",
    model_path: str = "artifacts/models/sklearn_pipeline_mlflow.joblib",
    feature_names_path: str = "artifacts/models/feature_names.json",
    risk_percentile: float = 50,
    treatment_effect_rel: float = 0.15,
    data_dir: str = "data/experiments",
    random_seed: int = SIMULATOR_SEED,
) -> Dict:
    """
    Run retention simulation using existing churn model and data.
    
    Selects top risk_percentile% high-risk customers, assigns treatment/control,
    simulates outcomes.
    
    Returns:
        Run summary dict
    """
    import json
    
    df = pd.read_csv(data_path)
    
    model = load_churn_model(model_path)
    
    with open(feature_names_path) as f:
        feat_info = json.load(f)
    
    feature_cols = feat_info.get("numeric_features", []) + feat_info.get(
        "categorical_features", []
    )
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in ["customerID", "Churn", "ChurnLabel"]]
    
    churn_probs = predict_churn_probs(model, df, feature_cols)
    
    threshold = np.percentile(churn_probs, 100 - risk_percentile)
    high_risk = churn_probs >= threshold
    df_hr = df[high_risk].copy()
    probs_hr = churn_probs[high_risk]
    
    if "customerID" not in df_hr.columns:
        df_hr["customerID"] = [f"CUST_{i}" for i in range(len(df_hr))]
    
    config = ExperimentConfig(
        experiment_id=experiment_id,
        name=f"Retention campaign - top {risk_percentile}% risk",
        allocation=0.5,
    )
    
    segment_cols = []
    for c in ["tenure", "Contract", "InternetService"]:
        if c in df_hr.columns:
            segment_cols.append(c)
            break
    
    return run_retention_simulation(
        experiment_id=experiment_id,
        customers_df=df_hr,
        churn_probs=probs_hr,
        config=config,
        treatment_effect_rel=treatment_effect_rel,
        segment_cols=segment_cols if segment_cols else None,
        data_dir=data_dir,
        random_seed=random_seed,
    )


def run_replay(
    experiment_id: str,
    data_path: str = "data/raw/Telco-Customer-Churn.csv",
    treatment_effect_rel: float = 0.15,
    max_customers: int = 2000,
    entity_id_col: str = "customerID",
    churn_col: str = "Churn",
    data_dir: str = "data/experiments",
    random_seed: int = SIMULATOR_SEED,
) -> Dict:
    """
    Replay pipeline: use Kaggle churn label as baseline outcome, simulate treatment.
    
    No model required. Assigns treatment/control, then for treatment arm
    artificially reduces churn by treatment_effect_rel (flips some churners to non-churn).
    """
    np.random.seed(random_seed)
    
    df = pd.read_csv(data_path, nrows=max_customers * 2)
    if len(df) > max_customers:
        df = df.sample(n=max_customers, random_state=random_seed)
    
    if entity_id_col not in df.columns:
        df[entity_id_col] = [f"CUST_{i}" for i in range(len(df))]
    
    churn_raw = df[churn_col].map(lambda x: 1 if str(x).strip().lower() in ("yes", "1", "true") else 0)
    
    config = ExperimentConfig(
        experiment_id=experiment_id,
        name=f"Replay experiment {experiment_id}",
        allocation=0.5,
    )
    
    entity_ids = df[entity_id_col].astype(str).tolist()
    assignments = assign_entities(entity_ids, config)
    assign_by_entity = {a.entity_id: a for a in assignments}
    
    now = datetime.utcnow()
    exposures = []
    outcomes = []
    
    for j, (i, row) in enumerate(df.iterrows()):
        eid = str(row[entity_id_col])
        a = assign_by_entity.get(eid)
        if not a:
            continue
        
        base_churned = churn_raw.iloc[j]
        
        if a.arm == ArmType.TREATMENT and base_churned == 1:
            flip = np.random.random() < treatment_effect_rel
            churned = 0 if flip else 1
        else:
            churned = base_churned
        
        exposures.append(ExposureEvent(
            experiment_id=experiment_id,
            entity_id=eid,
            arm=a.arm,
            exposed_at=now,
        ))
        outcomes.append(OutcomeEvent(
            experiment_id=experiment_id,
            entity_id=eid,
            arm=a.arm,
            outcome_name="churned",
            outcome_value=float(churned),
            observed_at=now,
        ))
    
    parts_data = []
    for e, o in zip(exposures, outcomes):
        row = {"entity_id": e.entity_id, "arm": e.arm.value, "outcome_value": o.outcome_value}
        mask = df[entity_id_col].astype(str) == e.entity_id
        if mask.any():
            cust_row = df[mask].iloc[0]
            for c in ["tenure", "Contract", "InternetService"]:
                if c in cust_row:
                    row[c] = cust_row[c]
        parts_data.append(row)
    write_participants(pd.DataFrame(parts_data), experiment_id, base_dir=data_dir)
    
    n_exp = append_exposures(exposures, experiment_id, base_dir=data_dir)
    n_out = append_outcomes(outcomes, experiment_id, base_dir=data_dir)
    
    n_control = sum(1 for a in assignments if a.arm == ArmType.CONTROL)
    n_treatment = sum(1 for a in assignments if a.arm == ArmType.TREATMENT)
    
    return {
        "experiment_id": experiment_id,
        "n_assigned": len(assignments),
        "n_control": n_control,
        "n_treatment": n_treatment,
        "exposures_written": n_exp,
        "outcomes_written": n_out,
        "treatment_effect_rel": treatment_effect_rel,
        "mode": "replay",
        "random_seed": random_seed,
    }
