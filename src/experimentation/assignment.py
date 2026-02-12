"""
Deterministic experiment assignment for A/B testing.

Uses hashing of (entity_id, experiment_id) to ensure stable assignments
with configurable allocation and optional stratification by risk bucket.
"""

import hashlib
import logging
from typing import List, Optional

from .schema import ArmType, Assignment, ExperimentConfig

logger = logging.getLogger(__name__)


def _hash_to_bucket(entity_id: str, experiment_id: str, salt: str = "") -> int:
    """
    Deterministic hash to [0, 9999] bucket.
    
    Same entity + experiment_id always maps to same bucket.
    """
    key = f"{entity_id}:{experiment_id}:{salt}"
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:8], 16) % 10000


def assign_entity(
    entity_id: str,
    experiment_id: str,
    allocation: float = 0.5,
    stratum: Optional[str] = None,
) -> ArmType:
    """
    Assign entity to control or treatment deterministically.
    
    Args:
        entity_id: Unique identifier (e.g., customerID)
        experiment_id: Experiment identifier
        allocation: Fraction in treatment (0.5 = 50/50)
        stratum: Optional stratum for stratification (used in hash for consistency)
        
    Returns:
        ArmType: CONTROL or TREATMENT
    """
    salt = stratum or ""
    bucket = _hash_to_bucket(entity_id, experiment_id, salt)
    threshold = int(allocation * 10000)
    return ArmType.TREATMENT if bucket < threshold else ArmType.CONTROL


def assign_entities(
    entity_ids: List[str],
    config: ExperimentConfig,
    stratify_by: Optional[dict] = None,
) -> List[Assignment]:
    """
    Assign multiple entities to experiment arms.
    
    Args:
        entity_ids: List of entity identifiers
        config: Experiment configuration
        stratify_by: Optional dict mapping entity_id -> stratum value.
                     When provided, allocation is balanced within strata.
        
    Returns:
        List of Assignment objects
    """
    from datetime import datetime
    assignments = []
    
    for eid in entity_ids:
        stratum = None
        if stratify_by and eid in stratify_by:
            stratum = str(stratify_by[eid])
        
        arm = assign_entity(
            entity_id=eid,
            experiment_id=config.experiment_id,
            allocation=config.allocation,
            stratum=stratum,
        )
        assignments.append(Assignment(
            experiment_id=config.experiment_id,
            entity_id=eid,
            arm=arm,
            assigned_at=datetime.utcnow(),
            stratum=stratum,
        ))
    
    n_control = sum(1 for a in assignments if a.arm == ArmType.CONTROL)
    n_treatment = sum(1 for a in assignments if a.arm == ArmType.TREATMENT)
    logger.info(
        f"Assignment complete: {len(assignments)} entities -> "
        f"control={n_control}, treatment={n_treatment}"
    )
    return assignments
