"""HoneyBee_v2 QC audit engine.

Audits a contributor's side-by-side evaluation work product against the 21 custom
dimensions of project 6a70ffe56999de9083413f7d.

Two things are decisive and shape the whole package:

1. Auditor independence. Almost every dimension is a disagreement measurement, so
   judgments are formed before the contributor's values enter context. Anchoring
   does not merely add noise; it biases every rate toward agreement.
2. Deterministic arithmetic. Models find issues; Python counts them. Every
   threshold here is a count or a percentage and none is produced by a model.
"""

from .config import DEFAULT_POLICY, POLICY_VERSION, Policy
from .errors import ERROR_CODES, error_code
from .registry import BLOCKS, CENSUS_GATES, ORDER, REGISTRY, CheckSpec
from .scoring import CheckVerdict, Measurement, ShapeViolation, build_verdict

__all__ = [
    "DEFAULT_POLICY",
    "POLICY_VERSION",
    "Policy",
    "ERROR_CODES",
    "error_code",
    "REGISTRY",
    "ORDER",
    "BLOCKS",
    "CENSUS_GATES",
    "CheckSpec",
    "CheckVerdict",
    "Measurement",
    "ShapeViolation",
    "build_verdict",
]
