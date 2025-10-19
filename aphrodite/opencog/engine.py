"""OpenCog Engine integration with Aphrodite.

This module provides the main OpenCogEngine class that integrates OpenCog's
cognitive architecture with Aphrodite's LLM inference engine for distributed
cognition at scale.
"""

from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
import time
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aphrodite.opencog.atomspace import (AtomSpace, Atom, AtomType, 
                                             TruthValue)
    from aphrodite.opencog.config import OpenCogConfig, CognitiveMode
    from aphrodite.opencog.distributed import (DistributedCognitionCoordinator,
                                               CognitiveTask, CognitiveResult)
    from aphrodite.common.sequence import SequenceGroup, SequenceGroupOutput
    from aphrodite.common.outputs import RequestOutput
else:
    # Avoid circular imports
    AtomSpace = Any
    Atom = Any
    AtomType = Any
    TruthValue = Any
    OpenCogConfig = Any
    CognitiveMode = Any
    DistributedCognitionCoordinator = Any
    CognitiveTask = Any
    CognitiveResult = Any
    SequenceGroup = Any
    SequenceGroupOutput = Any
    RequestOutput = Any


class OpenCogEngine:
    """OpenCog Engine for distributed cognition over LLM inference.
    
    This engine integrates OpenCog's cognitive architecture with Aphrodite's
    LLM inference engine, providing:
    - Knowledge representation via AtomSpace
    - Distributed cognitive processing
    - Semantic grounding of LLM outputs
    - Attention allocation and focus
    - Cognitive inference and reasoning
    """
    
    def __init__(self, config: Optional['OpenCogConfig'] = None):
        """Initialize the OpenCog Engine.
        
        Args:
            config: OpenCog configuration. If None, uses default config.
        """
        # Runtime imports to avoid circular dependencies
        from aphrodite.opencog.atomspace import AtomSpace, AtomType
        from aphrodite.opencog.config import OpenCogConfig as ConfigClass
        from aphrodite.opencog.distributed import DistributedCognitionCoordinator
        
        self.config = config or ConfigClass()
        self.config.validate()
        
        # Initialize main AtomSpace
        self.atomspace = AtomSpace()
        
        # Initialize distributed coordinator if enabled
        self.coordinator: Optional[DistributedCognitionCoordinator] = None
        if self.config.distributed_config.enable_distributed:
            self.coordinator = DistributedCognitionCoordinator(
                self.config,
                self.config.distributed_config.num_cognitive_workers
            )
        
        # Cognitive cache for frequent patterns
        self._cognitive_cache: Dict[str, Any] = {}
        
        # Metrics
        self._metrics = {
            "total_cognitive_operations": 0,
            "total_atoms_created": 0,
            "total_inferences": 0,
            "total_processing_time_ms": 0.0,
        }
        
        logger.info(f"OpenCogEngine initialized with mode "
                   f"{self.config.cognitive_mode.value}")
    
    def ground_text(self, text: str, context: Optional[Dict[str, Any]] = None
                   ) -> List['Atom']:
        """Ground text in the AtomSpace by creating semantic representations.
        
        Args:
            text: Text to ground
            context: Optional context for grounding
            
        Returns:
            List of atoms created for the text
        """
        from aphrodite.opencog.atomspace import AtomType, TruthValue
        
        start_time = time.time()
        atoms = []
        
        # Create document node
        doc_node = self.atomspace.add_node(
            AtomType.DOCUMENT_NODE,
            f"doc_{hash(text)}",
            TruthValue(strength=1.0, confidence=0.8)
        )
        atoms.append(doc_node)
        
        # Parse text into sentences (simple split for now)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        for i, sentence in enumerate(sentences):
            # Create sentence node
            sent_node = self.atomspace.add_node(
                AtomType.SENTENCE_NODE,
                f"sent_{hash(sentence)}",
                TruthValue(strength=1.0, confidence=0.8)
            )
            atoms.append(sent_node)
            
            # Link sentence to document
            member_link = self.atomspace.add_link(
                AtomType.MEMBER_LINK,
                [sent_node, doc_node],
                TruthValue(strength=1.0, confidence=1.0)
            )
            atoms.append(member_link)
            
            # Parse words (simple split)
            words = sentence.split()
            for word in words:
                word = word.lower().strip(',.!?;:')
                if not word:
                    continue
                
                # Create word node
                word_node = self.atomspace.add_node(
                    AtomType.WORD_NODE,
                    word,
                    TruthValue(strength=1.0, confidence=0.9)
                )
                atoms.append(word_node)
                
                # Link word to sentence
                word_link = self.atomspace.add_link(
                    AtomType.MEMBER_LINK,
                    [word_node, sent_node],
                    TruthValue(strength=1.0, confidence=1.0)
                )
                atoms.append(word_link)
        
        # Add context if provided
        if context:
            for key, value in context.items():
                ctx_node = self.atomspace.add_node(
                    AtomType.CONCEPT_NODE,
                    f"ctx_{key}_{value}",
                    TruthValue(strength=0.9, confidence=0.7)
                )
                atoms.append(ctx_node)
                
                ctx_link = self.atomspace.add_link(
                    AtomType.CONTEXT_LINK,
                    [doc_node, ctx_node],
                    TruthValue(strength=0.9, confidence=0.8)
                )
                atoms.append(ctx_link)
        
        # Update metrics
        elapsed = (time.time() - start_time) * 1000
        self._metrics["total_cognitive_operations"] += 1
        self._metrics["total_atoms_created"] += len(atoms)
        self._metrics["total_processing_time_ms"] += elapsed
        
        logger.debug(f"Grounded text into {len(atoms)} atoms in {elapsed:.2f}ms")
        return atoms
    
    def process_sequence_group(self, 
                               seq_group: 'SequenceGroup',
                               output: Optional['SequenceGroupOutput'] = None
                              ) -> Optional['CognitiveResult']:
        """Process a sequence group through cognitive architecture.
        
        Args:
            seq_group: Sequence group from Aphrodite engine
            output: Optional output from the sequence group
            
        Returns:
            Cognitive processing result if applicable
        """
        from aphrodite.opencog.distributed import CognitiveTask, CognitiveResult
        from aphrodite.opencog.config import CognitiveMode
        
        if not self.config.enable_opencog:
            return None
        
        start_time = time.time()
        
        # Extract prompt text
        prompt = seq_group.prompt
        if not prompt:
            return None
        
        # Ground prompt in AtomSpace
        atoms = self.ground_text(prompt)
        
        # Create cognitive task
        task = CognitiveTask(
            task_id=f"seq_{seq_group.request_id}",
            task_type="sequence_processing",
            atoms=atoms,
            priority=1.0,
            metadata={"request_id": seq_group.request_id}
        )
        
        # Process based on mode
        if self.config.cognitive_mode == CognitiveMode.REACTIVE:
            # Fast, immediate processing
            result = self._process_reactive(task)
        elif self.config.cognitive_mode == CognitiveMode.DELIBERATIVE:
            # Slower, thoughtful processing
            result = self._process_deliberative(task)
        else:  # HYBRID
            # Adaptive processing
            result = self._process_hybrid(task)
        
        elapsed = (time.time() - start_time) * 1000
        
        # Check latency constraint
        if elapsed > self.config.max_cognitive_latency_ms:
            logger.warning(f"Cognitive processing exceeded latency limit: "
                         f"{elapsed:.2f}ms > "
                         f"{self.config.max_cognitive_latency_ms}ms")
        
        return result
    
    def _process_reactive(self, task: 'CognitiveTask') -> 'CognitiveResult':
        """Process task in reactive mode (fast)."""
        from aphrodite.opencog.distributed import CognitiveResult
        
        # Simple local processing without inference
        return CognitiveResult(
            task_id=task.task_id,
            success=True,
            atoms=task.atoms,
            inferences=[],
            confidence=1.0,
            metadata={"mode": "reactive"}
        )
    
    def _process_deliberative(self, task: 'CognitiveTask') -> 'CognitiveResult':
        """Process task in deliberative mode (thorough)."""
        from aphrodite.opencog.distributed import CognitiveResult
        
        if self.coordinator:
            # Use distributed processing
            self.coordinator.submit_task(task)
            self.coordinator.process_tasks()
            result = self.coordinator.get_result(task.task_id)
            if result:
                result.metadata["mode"] = "deliberative"
                return result
        
        # Fallback to local processing
        return CognitiveResult(
            task_id=task.task_id,
            success=True,
            atoms=task.atoms,
            inferences=[],
            confidence=0.8,
            metadata={"mode": "deliberative_local"}
        )
    
    def _process_hybrid(self, task: 'CognitiveTask') -> 'CognitiveResult':
        """Process task in hybrid mode (adaptive)."""
        # Decide based on task complexity
        if len(task.atoms) < 10:
            return self._process_reactive(task)
        else:
            return self._process_deliberative(task)
    
    def get_focused_atoms(self, limit: Optional[int] = None) -> List['Atom']:
        """Get atoms in current attentional focus.
        
        Args:
            limit: Maximum number of atoms to return
            
        Returns:
            List of atoms with high attention values
        """
        from aphrodite.opencog.atomspace import AtomType
        
        if not self.config.attention_config.enable_attention_allocation:
            return []
        
        max_atoms = (limit or 
                    self.config.attention_config.max_focus_atoms)
        
        # Get all atoms and sort by STI
        all_atoms = []
        for atom_type in AtomType:
            all_atoms.extend(self.atomspace.get_atoms_by_type(atom_type))
        
        # Sort by short-term importance
        focused = sorted(
            all_atoms,
            key=lambda a: a.attention_value.sti,
            reverse=True
        )[:max_atoms]
        
        return focused
    
    def allocate_attention(self, atoms: List['Atom'], 
                          importance: float = 1.0) -> None:
        """Allocate attention to specific atoms.
        
        Args:
            atoms: Atoms to focus attention on
            importance: Importance level [0, 1]
        """
        if not self.config.attention_config.enable_attention_allocation:
            return
        
        for atom in atoms:
            # Increase STI
            atom.attention_value.sti += importance
            
            # Cap at reasonable limit
            atom.attention_value.sti = min(atom.attention_value.sti, 10.0)
    
    def decay_attention(self) -> None:
        """Decay attention values over time."""
        from aphrodite.opencog.atomspace import AtomType
        
        if not self.config.attention_config.enable_attention_allocation:
            return
        
        decay_rate = self.config.attention_config.attention_decay_rate
        
        all_atoms = []
        for atom_type in AtomType:
            all_atoms.extend(self.atomspace.get_atoms_by_type(atom_type))
        
        for atom in all_atoms:
            atom.attention_value.sti *= decay_rate
            
            # Remove from focus if below threshold
            if atom.attention_value.sti < 0.01:
                atom.attention_value.sti = 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cognitive processing metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = self._metrics.copy()
        metrics["atomspace_size"] = self.atomspace.size()
        
        if self.coordinator:
            metrics["distributed"] = self.coordinator.get_statistics()
        
        return metrics
    
    def synchronize(self) -> None:
        """Synchronize distributed components."""
        if self.coordinator:
            self.coordinator.synchronize_atomspaces()
    
    def reset(self) -> None:
        """Reset the cognitive state."""
        self.atomspace.clear()
        self._cognitive_cache.clear()
        
        self._metrics = {
            "total_cognitive_operations": 0,
            "total_atoms_created": 0,
            "total_inferences": 0,
            "total_processing_time_ms": 0.0,
        }
        
        logger.info("OpenCogEngine reset")
    
    def __str__(self):
        return (f"OpenCogEngine(mode={self.config.cognitive_mode.value}, "
                f"atoms={self.atomspace.size()})")
    
    def __repr__(self):
        return self.__str__()
