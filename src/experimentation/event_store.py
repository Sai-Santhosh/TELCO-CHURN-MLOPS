"""
Lightweight event store for experiment exposures and outcomes.

Writes parquet (or csv fallback) to data/experiments/. Provides functions to append
exposures/outcomes and read by time window. Kafka-ready interfaces.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .schema import ArmType, ExposureEvent, OutcomeEvent

logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = "data/experiments"

try:
    import pyarrow
    _USE_PARQUET = True
except ImportError:
    _USE_PARQUET = False


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _exposures_path(experiment_id: str, base_dir: str = DEFAULT_STORE_DIR) -> Path:
    ext = "parquet" if _USE_PARQUET else "csv"
    return Path(base_dir) / experiment_id / f"exposures.{ext}"


def _outcomes_path(experiment_id: str, base_dir: str = DEFAULT_STORE_DIR) -> Path:
    ext = "parquet" if _USE_PARQUET else "csv"
    return Path(base_dir) / experiment_id / f"outcomes.{ext}"


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _write_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _event_to_row(evt: ExposureEvent) -> dict:
    return {
        "experiment_id": evt.experiment_id,
        "entity_id": evt.entity_id,
        "arm": evt.arm.value,
        "exposed_at": evt.exposed_at,
        "metadata": str(evt.metadata) if evt.metadata else "",
    }


def _outcome_to_row(evt: OutcomeEvent) -> dict:
    return {
        "experiment_id": evt.experiment_id,
        "entity_id": evt.entity_id,
        "arm": evt.arm.value,
        "outcome_name": evt.outcome_name,
        "outcome_value": evt.outcome_value,
        "observed_at": evt.observed_at,
        "metadata": str(evt.metadata) if evt.metadata else "",
    }


def append_exposures(
    events: List[ExposureEvent],
    experiment_id: str,
    base_dir: str = DEFAULT_STORE_DIR,
) -> int:
    """
    Append exposure events to the store.
    
    Args:
        events: List of ExposureEvent
        experiment_id: Experiment identifier
        base_dir: Base directory for experiment data
        
    Returns:
        Number of events appended
    """
    path = _exposures_path(experiment_id, base_dir)
    _ensure_dir(path.parent)
    
    df_new = pd.DataFrame([_event_to_row(e) for e in events])
    
    if path.exists():
        df_existing = _read_table(path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    
    _write_table(df, path)
    logger.info(f"Appended {len(events)} exposures to {path}")
    return len(events)


def append_outcomes(
    events: List[OutcomeEvent],
    experiment_id: str,
    base_dir: str = DEFAULT_STORE_DIR,
) -> int:
    """
    Append outcome events to the store.
    
    Args:
        events: List of OutcomeEvent
        experiment_id: Experiment identifier
        base_dir: Base directory for experiment data
        
    Returns:
        Number of events appended
    """
    path = _outcomes_path(experiment_id, base_dir)
    _ensure_dir(path.parent)
    
    df_new = pd.DataFrame([_outcome_to_row(e) for e in events])
    
    if path.exists():
        df_existing = _read_table(path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    
    _write_table(df, path)
    logger.info(f"Appended {len(events)} outcomes to {path}")
    return len(events)


def read_exposures(
    experiment_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    base_dir: str = DEFAULT_STORE_DIR,
) -> pd.DataFrame:
    """
    Read exposure events for an experiment, optionally filtered by time.
    
    Args:
        experiment_id: Experiment identifier
        start_date: Optional start of time window
        end_date: Optional end of time window
        base_dir: Base directory for experiment data
        
    Returns:
        DataFrame with exposure events
    """
    path = _exposures_path(experiment_id, base_dir)
    df = _read_table(path)
    if "exposed_at" in df.columns:
        df["exposed_at"] = pd.to_datetime(df["exposed_at"])
        if start_date:
            df = df[df["exposed_at"] >= start_date]
        if end_date:
            df = df[df["exposed_at"] <= end_date]
    return df


def read_outcomes(
    experiment_id: str,
    outcome_name: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    base_dir: str = DEFAULT_STORE_DIR,
) -> pd.DataFrame:
    """
    Read outcome events for an experiment.
    
    Args:
        experiment_id: Experiment identifier
        outcome_name: Optional filter by outcome name
        start_date: Optional start of time window
        end_date: Optional end of time window
        base_dir: Base directory for experiment data
        
    Returns:
        DataFrame with outcome events
    """
    path = _outcomes_path(experiment_id, base_dir)
    df = _read_table(path)
    if "observed_at" in df.columns:
        df["observed_at"] = pd.to_datetime(df["observed_at"])
        if start_date:
            df = df[df["observed_at"] >= start_date]
        if end_date:
            df = df[df["observed_at"] <= end_date]
    if outcome_name:
        df = df[df["outcome_name"] == outcome_name]
    return df


def write_participants(
    df: pd.DataFrame,
    experiment_id: str,
    base_dir: str = DEFAULT_STORE_DIR,
) -> None:
    """
    Write participant-level analysis table (entity_id, arm, outcome_value, segment cols).
    Enables direct analysis without merging exposures + outcomes.
    """
    path = Path(base_dir) / experiment_id / "participants.csv"
    _ensure_dir(path.parent)
    df.to_csv(path, index=False)
    logger.info(f"Participants written to {path}")


def read_participants(
    experiment_id: str,
    base_dir: str = DEFAULT_STORE_DIR,
) -> pd.DataFrame:
    """Read participant-level table if available."""
    path = Path(base_dir) / experiment_id / "participants.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_experiment_summary(experiment_id: str, base_dir: str = DEFAULT_STORE_DIR) -> dict:
    """
    Get summary counts for an experiment.
    
    Returns:
        Dict with n_exposures, n_outcomes, control_count, treatment_count
    """
    exp_df = read_exposures(experiment_id, base_dir=base_dir)
    out_df = read_outcomes(experiment_id, base_dir=base_dir)
    
    n_control = 0
    n_treatment = 0
    if not exp_df.empty and "arm" in exp_df.columns:
        n_control = (exp_df["arm"] == ArmType.CONTROL.value).sum()
        n_treatment = (exp_df["arm"] == ArmType.TREATMENT.value).sum()
    
    return {
        "n_exposures": len(exp_df),
        "n_outcomes": len(out_df),
        "control_count": n_control,
        "treatment_count": n_treatment,
    }
