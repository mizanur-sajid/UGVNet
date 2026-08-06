"""UGVNet: Unified Global Vision Network."""

from .data_audit import (
    DatasetAuditError,
    audit_dataset,
    enforce_audit_policy,
)
from .hybrid import UGVNetHybrid, ugvnet_hybrid
from .lightweight import (
    UGVNet,
    UGVNetConfig,
    create_ugvnet,
    ugvnet_base,
    ugvnet_small,
    ugvnet_tiny,
)

__all__ = [
    "DatasetAuditError",
    "UGVNet",
    "UGVNetConfig",
    "UGVNetHybrid",
    "audit_dataset",
    "create_ugvnet",
    "enforce_audit_policy",
    "ugvnet_base",
    "ugvnet_hybrid",
    "ugvnet_small",
    "ugvnet_tiny",
]

__version__ = "0.2.0"
