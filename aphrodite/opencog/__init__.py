"""OpenCog integration module for Aphrodite Engine.

This module provides distributed cognition capabilities for large-scale LLM 
inference by integrating OpenCog's cognitive architecture with Aphrodite's 
inference engine.
"""

from aphrodite.opencog.atomspace import AtomSpace, AtomType
from aphrodite.opencog.config import OpenCogConfig
from aphrodite.opencog.engine import OpenCogEngine
from aphrodite.opencog.integration import (OpenCogIntegration, 
                                           create_opencog_integration)

__all__ = [
    "AtomSpace",
    "AtomType",
    "OpenCogConfig",
    "OpenCogEngine",
    "OpenCogIntegration",
    "create_opencog_integration",
]
