"""Integration utilities for connecting OpenCog with Aphrodite Engine.

This module provides utilities to integrate OpenCog cognitive processing
with Aphrodite's LLM inference pipeline.
"""

from typing import Optional, TYPE_CHECKING
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aphrodite.opencog.engine import OpenCogEngine
    from aphrodite.opencog.config import OpenCogConfig
    from aphrodite.common.sequence import SequenceGroup
    from aphrodite.common.outputs import RequestOutput


class OpenCogIntegration:
    """Integration layer between OpenCog and Aphrodite Engine.
    
    This class provides hooks to process sequences through the OpenCog
    cognitive architecture during LLM inference.
    """
    
    def __init__(self, config: Optional['OpenCogConfig'] = None):
        """Initialize OpenCog integration.
        
        Args:
            config: OpenCog configuration. If None, integration is disabled.
        """
        self.enabled = config is not None
        self.opencog_engine: Optional['OpenCogEngine'] = None
        
        if self.enabled:
            try:
                # Lazy import to avoid circular dependencies
                from aphrodite.opencog.engine import OpenCogEngine
                
                self.opencog_engine = OpenCogEngine(config)
                logger.info("OpenCog integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenCog: {e}")
                self.enabled = False
    
    def process_prompt(self, prompt: str, 
                      request_id: Optional[str] = None) -> Optional[dict]:
        """Process a prompt through OpenCog cognitive architecture.
        
        Args:
            prompt: The prompt text to process
            request_id: Optional request ID for tracking
            
        Returns:
            Dictionary with cognitive processing results, or None if disabled
        """
        if not self.enabled or not self.opencog_engine:
            return None
        
        try:
            # Ground prompt in AtomSpace
            atoms = self.opencog_engine.ground_text(prompt)
            
            # Get focused atoms
            focused = self.opencog_engine.get_focused_atoms(limit=10)
            
            return {
                "atoms_created": len(atoms),
                "focused_atoms": len(focused),
                "atomspace_size": self.opencog_engine.atomspace.size(),
            }
        except Exception as e:
            logger.debug(f"OpenCog processing error: {e}")
            return None
    
    def process_output(self, output_text: str,
                      request_id: Optional[str] = None) -> Optional[dict]:
        """Process model output through OpenCog.
        
        Args:
            output_text: The generated text to process
            request_id: Optional request ID for tracking
            
        Returns:
            Dictionary with cognitive processing results, or None if disabled
        """
        if not self.enabled or not self.opencog_engine:
            return None
        
        try:
            # Ground output in AtomSpace
            atoms = self.opencog_engine.ground_text(output_text)
            
            # Allocate attention to output atoms
            from aphrodite.opencog.atomspace import AtomType
            important_atoms = [a for a in atoms 
                             if a.atom_type == AtomType.CONCEPT_NODE][:5]
            if important_atoms:
                self.opencog_engine.allocate_attention(
                    important_atoms, importance=1.5)
            
            return {
                "atoms_created": len(atoms),
                "atomspace_size": self.opencog_engine.atomspace.size(),
            }
        except Exception as e:
            logger.debug(f"OpenCog processing error: {e}")
            return None
    
    def get_metrics(self) -> Optional[dict]:
        """Get cognitive processing metrics.
        
        Returns:
            Dictionary of metrics, or None if disabled
        """
        if not self.enabled or not self.opencog_engine:
            return None
        
        try:
            return self.opencog_engine.get_metrics()
        except Exception as e:
            logger.debug(f"Error getting metrics: {e}")
            return None
    
    def synchronize(self) -> None:
        """Synchronize distributed OpenCog components."""
        if self.enabled and self.opencog_engine:
            try:
                self.opencog_engine.synchronize()
            except Exception as e:
                logger.debug(f"Synchronization error: {e}")
    
    def reset(self) -> None:
        """Reset cognitive state."""
        if self.enabled and self.opencog_engine:
            try:
                self.opencog_engine.reset()
            except Exception as e:
                logger.debug(f"Reset error: {e}")


def create_opencog_integration(
    enable: bool = False,
    config_dict: Optional[dict] = None
) -> OpenCogIntegration:
    """Factory function to create OpenCog integration.
    
    Args:
        enable: Whether to enable OpenCog integration
        config_dict: Optional configuration dictionary
        
    Returns:
        OpenCogIntegration instance
    """
    if not enable:
        return OpenCogIntegration(config=None)
    
    try:
        from aphrodite.opencog.config import OpenCogConfig
        
        if config_dict:
            config = OpenCogConfig.from_dict(config_dict)
        else:
            config = OpenCogConfig(enable_opencog=True)
        
        return OpenCogIntegration(config=config)
    except Exception as e:
        logger.warning(f"Failed to create OpenCog integration: {e}")
        return OpenCogIntegration(config=None)
