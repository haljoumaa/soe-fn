"""Canonical units and frame conventions for SOE contracts."""

LENGTH_UNIT = "cm"
# Used only for invariant checks / floating roundoff; VOI containment remains exact half-open.
EPS_BOUNDARY = 1e-12

__all__ = ["LENGTH_UNIT", "EPS_BOUNDARY"]
