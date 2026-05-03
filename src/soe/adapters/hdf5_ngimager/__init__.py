"""Public interface for the NGImager HDF5 adapter.

`load_canonical_events(...)` is the authoritative canonical event-ingestion
boundary. `load_meta_and_cones(...)` remains a low-level reader helper for
legacy callers; reconstruction input assembly does not use HDF5 `/meta` as
VOI authority.
"""

from .reader import load_canonical_events, load_meta_and_cones
from .types import Cones, MetaGeometry, RawEventRecord

__all__ = [
    "Cones",
    "MetaGeometry",
    "RawEventRecord",
    "load_canonical_events",
    "load_meta_and_cones",
]
