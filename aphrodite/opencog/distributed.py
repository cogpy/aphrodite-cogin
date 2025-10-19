"""Distributed cognition coordinator for OpenCog integration.

This module coordinates cognitive processing across multiple workers
in a distributed Aphrodite deployment.
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING
import threading
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aphrodite.opencog.atomspace import AtomSpace, Atom, AtomType
    from aphrodite.opencog.config import OpenCogConfig, DistributionStrategy
else:
    # Avoid circular imports by using late binding
    AtomSpace = Any
    Atom = Any
    AtomType = Any
    OpenCogConfig = Any
    DistributionStrategy = Any


@dataclass
class CognitiveTask:
    """A task for distributed cognitive processing."""
    task_id: str
    task_type: str
    atoms: List[Any]  # List[Atom]
    priority: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CognitiveResult:
    """Result from cognitive processing."""
    task_id: str
    success: bool
    atoms: List[Any]  # List[Atom]
    inferences: List[Any]  # List[Atom]
    confidence: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CognitiveWorker:
    """Worker for processing cognitive tasks."""
    
    def __init__(self, worker_id: int, atomspace: 'AtomSpace',
                 config: 'OpenCogConfig'):
        self.worker_id = worker_id
        self.atomspace = atomspace
        self.config = config
        self.is_running = False
        self._lock = threading.Lock()
        logger.info(f"Cognitive worker {worker_id} initialized")
    
    def process_task(self, task: CognitiveTask) -> CognitiveResult:
        """Process a cognitive task.
        
        Args:
            task: The cognitive task to process
            
        Returns:
            Result of the cognitive processing
        """
        from aphrodite.opencog.atomspace import AtomType
        
        with self._lock:
            logger.debug(f"Worker {self.worker_id} processing task "
                        f"{task.task_id}")
            
            # Add atoms to local AtomSpace
            for atom in task.atoms:
                if atom.atom_type.name.endswith('_NODE'):
                    self.atomspace.add_node(
                        atom.atom_type, atom.name, atom.truth_value)
                else:
                    self.atomspace.add_link(
                        atom.atom_type, atom.outgoing, atom.truth_value)
            
            # Perform cognitive processing
            inferences = self._perform_inference(task.atoms)
            
            result = CognitiveResult(
                task_id=task.task_id,
                success=True,
                atoms=task.atoms,
                inferences=inferences,
                confidence=self._calculate_confidence(inferences),
                metadata={"worker_id": self.worker_id}
            )
            
            logger.debug(f"Worker {self.worker_id} completed task "
                        f"{task.task_id}")
            return result
    
    def _perform_inference(self, atoms: List[Atom]) -> List[Atom]:
        """Perform inference on atoms."""
        inferences = []
        
        if not self.config.inference_config.enable_inference:
            return inferences
        
        # Simple pattern matching inference
        if self.config.inference_config.pattern_matching_enabled:
            for atom in atoms:
                if atom.atom_type == AtomType.CONCEPT_NODE:
                    # Find related concepts
                    related = self.atomspace.get_incoming(atom)
                    for rel in related:
                        if (rel.atom_type == AtomType.SIMILARITY_LINK and
                            rel.truth_value.strength > 
                            self.config.inference_config.
                            inference_confidence_threshold):
                            inferences.append(rel)
        
        return inferences
    
    def _calculate_confidence(self, inferences: List[Atom]) -> float:
        """Calculate overall confidence from inferences."""
        if not inferences:
            return 0.0
        
        total_strength = sum(inf.truth_value.strength for inf in inferences)
        return total_strength / len(inferences)


class DistributedCognitionCoordinator:
    """Coordinates cognitive processing across distributed workers.
    
    This coordinator manages the distribution of cognitive tasks across
    multiple workers and synchronizes their AtomSpaces.
    """
    
    def __init__(self, config: 'OpenCogConfig', 
                 num_workers: Optional[int] = None):
        from aphrodite.opencog.atomspace import AtomSpace
        
        self.config = config
        self.num_workers = (num_workers or 
                          config.distributed_config.num_cognitive_workers or 
                          1)
        
        # Create AtomSpaces for each worker
        self.atomspaces: List[Any] = [  # List[AtomSpace]
            AtomSpace() for _ in range(self.num_workers)
        ]
        
        # Create workers
        self.workers: List[CognitiveWorker] = [
            CognitiveWorker(i, self.atomspaces[i], config)
            for i in range(self.num_workers)
        ]
        
        # Task queue
        self._task_queue: List[CognitiveTask] = []
        self._task_lock = threading.Lock()
        
        # Results
        self._results: Dict[str, CognitiveResult] = {}
        self._results_lock = threading.Lock()
        
        logger.info(f"DistributedCognitionCoordinator initialized with "
                   f"{self.num_workers} workers")
    
    def submit_task(self, task: CognitiveTask) -> str:
        """Submit a cognitive task for processing.
        
        Args:
            task: The cognitive task to submit
            
        Returns:
            Task ID for retrieving results
        """
        with self._task_lock:
            self._task_queue.append(task)
            logger.debug(f"Task {task.task_id} submitted")
            return task.task_id
    
    def get_result(self, task_id: str, 
                   timeout: Optional[float] = None) -> Optional[CognitiveResult]:
        """Get result for a task.
        
        Args:
            task_id: ID of the task
            timeout: Optional timeout in seconds
            
        Returns:
            Result if available, None otherwise
        """
        with self._results_lock:
            return self._results.get(task_id)
    
    def process_tasks(self) -> None:
        """Process all pending tasks."""
        with self._task_lock:
            if not self._task_queue:
                return
            
            tasks = self._task_queue.copy()
            self._task_queue.clear()
        
        # Distribute tasks based on strategy
        strategy = self.config.distributed_config.distribution_strategy
        
        if strategy == DistributionStrategy.DATA_PARALLEL:
            results = self._process_data_parallel(tasks)
        elif strategy == DistributionStrategy.TENSOR_PARALLEL:
            results = self._process_tensor_parallel(tasks)
        elif strategy == DistributionStrategy.PIPELINE_PARALLEL:
            results = self._process_pipeline_parallel(tasks)
        else:  # HYBRID
            results = self._process_hybrid(tasks)
        
        # Store results
        with self._results_lock:
            for result in results:
                self._results[result.task_id] = result
        
        logger.debug(f"Processed {len(results)} tasks")
    
    def _process_data_parallel(self, 
                               tasks: List[CognitiveTask]) -> List[CognitiveResult]:
        """Process tasks in data-parallel mode."""
        results = []
        for i, task in enumerate(tasks):
            worker_idx = i % self.num_workers
            result = self.workers[worker_idx].process_task(task)
            results.append(result)
        return results
    
    def _process_tensor_parallel(self, 
                                 tasks: List[CognitiveTask]) -> List[CognitiveResult]:
        """Process tasks in tensor-parallel mode."""
        # For tensor parallel, split atoms across workers
        results = []
        for task in tasks:
            atoms_per_worker = len(task.atoms) // self.num_workers
            worker_results = []
            
            for i in range(self.num_workers):
                start_idx = i * atoms_per_worker
                end_idx = (start_idx + atoms_per_worker 
                          if i < self.num_workers - 1 
                          else len(task.atoms))
                
                sub_task = CognitiveTask(
                    task_id=f"{task.task_id}_w{i}",
                    task_type=task.task_type,
                    atoms=task.atoms[start_idx:end_idx],
                    priority=task.priority,
                    metadata=task.metadata
                )
                worker_results.append(
                    self.workers[i].process_task(sub_task))
            
            # Merge results
            merged_result = self._merge_results(task.task_id, worker_results)
            results.append(merged_result)
        
        return results
    
    def _process_pipeline_parallel(self, 
                                    tasks: List[CognitiveTask]) -> List[CognitiveResult]:
        """Process tasks in pipeline-parallel mode."""
        # For pipeline parallel, process tasks through workers sequentially
        results = []
        for task in tasks:
            current_task = task
            for worker in self.workers:
                result = worker.process_task(current_task)
                # Use result atoms as input for next stage
                current_task = CognitiveTask(
                    task_id=current_task.task_id,
                    task_type=current_task.task_type,
                    atoms=result.atoms + result.inferences,
                    priority=current_task.priority,
                    metadata=current_task.metadata
                )
            results.append(result)
        return results
    
    def _process_hybrid(self, 
                        tasks: List[CognitiveTask]) -> List[CognitiveResult]:
        """Process tasks in hybrid mode."""
        # Use data parallel for simplicity in hybrid mode
        return self._process_data_parallel(tasks)
    
    def _merge_results(self, task_id: str, 
                      results: List[CognitiveResult]) -> CognitiveResult:
        """Merge results from multiple workers."""
        all_atoms = []
        all_inferences = []
        confidences = []
        
        for result in results:
            all_atoms.extend(result.atoms)
            all_inferences.extend(result.inferences)
            confidences.append(result.confidence)
        
        avg_confidence = (sum(confidences) / len(confidences) 
                         if confidences else 0.0)
        
        return CognitiveResult(
            task_id=task_id,
            success=True,
            atoms=all_atoms,
            inferences=all_inferences,
            confidence=avg_confidence,
            metadata={"merged": True, "num_workers": len(results)}
        )
    
    def synchronize_atomspaces(self) -> None:
        """Synchronize AtomSpaces across workers."""
        if not self.config.distributed_config.enable_knowledge_sharing:
            return
        
        logger.debug("Synchronizing AtomSpaces across workers")
        
        # Collect all unique atoms
        all_atoms = set()
        for atomspace in self.atomspaces:
            for atom_type in AtomType:
                atoms = atomspace.get_atoms_by_type(atom_type)
                all_atoms.update(atoms)
        
        # Distribute to all workers
        for atomspace in self.atomspaces:
            for atom in all_atoms:
                if atom.atom_type.name.endswith('_NODE'):
                    atomspace.add_node(
                        atom.atom_type, atom.name, atom.truth_value)
                else:
                    atomspace.add_link(
                        atom.atom_type, atom.outgoing, atom.truth_value)
        
        logger.debug(f"Synchronized {len(all_atoms)} atoms across workers")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about distributed cognition."""
        total_atoms = sum(as_.size() for as_ in self.atomspaces)
        
        return {
            "num_workers": self.num_workers,
            "total_atoms": total_atoms,
            "avg_atoms_per_worker": total_atoms / self.num_workers,
            "pending_tasks": len(self._task_queue),
            "completed_tasks": len(self._results),
            "distribution_strategy": 
                self.config.distributed_config.distribution_strategy.value,
        }
