"""AtomSpace implementation for OpenCog knowledge representation.

The AtomSpace is a hypergraph database that stores knowledge as atoms and
their relationships. This implementation provides a lightweight version
suitable for integration with Aphrodite's LLM inference engine.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
import threading
import logging

logger = logging.getLogger(__name__)


class AtomType(Enum):
    """Types of atoms in the AtomSpace."""
    # Node types
    CONCEPT_NODE = "ConceptNode"
    PREDICATE_NODE = "PredicateNode"
    SCHEMA_NODE = "SchemaNode"
    GROUNDED_SCHEMA_NODE = "GroundedSchemaNode"
    VARIABLE_NODE = "VariableNode"
    WORD_NODE = "WordNode"
    SENTENCE_NODE = "SentenceNode"
    DOCUMENT_NODE = "DocumentNode"
    
    # Link types
    INHERITANCE_LINK = "InheritanceLink"
    SIMILARITY_LINK = "SimilarityLink"
    EVALUATION_LINK = "EvaluationLink"
    EXECUTION_LINK = "ExecutionLink"
    LIST_LINK = "ListLink"
    MEMBER_LINK = "MemberLink"
    SUBSET_LINK = "SubsetLink"
    
    # Cognitive types
    CONTEXT_LINK = "ContextLink"
    INFERENCE_LINK = "InferenceLink"
    ATTENTION_VALUE = "AttentionValue"


@dataclass
class TruthValue:
    """Truth value with strength and confidence."""
    strength: float = 1.0  # [0, 1]
    confidence: float = 1.0  # [0, 1]
    
    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class AttentionValue:
    """Attention value for cognitive focus."""
    sti: float = 0.0  # Short-term importance
    lti: float = 0.0  # Long-term importance
    vlti: float = 0.0  # Very long-term importance


@dataclass
class Atom:
    """Base class for atoms in the AtomSpace."""
    atom_type: AtomType
    name: str
    truth_value: TruthValue = field(default_factory=TruthValue)
    attention_value: AttentionValue = field(default_factory=AttentionValue)
    outgoing: List['Atom'] = field(default_factory=list)
    incoming: Set['Atom'] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.atom_type, self.name, tuple(self.outgoing)))
    
    def __eq__(self, other):
        if not isinstance(other, Atom):
            return False
        return (self.atom_type == other.atom_type and 
                self.name == other.name and 
                self.outgoing == other.outgoing)


class AtomSpace:
    """Hypergraph database for storing cognitive knowledge.
    
    The AtomSpace provides a thread-safe storage and retrieval system for
    atoms and their relationships. It supports distributed operations and
    integrates with Aphrodite's inference engine.
    """
    
    def __init__(self):
        self._atoms: Dict[str, Atom] = {}
        self._lock = threading.RLock()
        self._atom_counter = 0
        logger.info("AtomSpace initialized")
    
    def add_node(self, 
                 atom_type: AtomType, 
                 name: str,
                 truth_value: Optional[TruthValue] = None) -> Atom:
        """Add a node to the AtomSpace.
        
        Args:
            atom_type: Type of the atom
            name: Name/identifier for the atom
            truth_value: Optional truth value
            
        Returns:
            The created or existing atom
        """
        with self._lock:
            key = self._make_key(atom_type, name, [])
            if key in self._atoms:
                return self._atoms[key]
            
            atom = Atom(
                atom_type=atom_type,
                name=name,
                truth_value=truth_value or TruthValue()
            )
            self._atoms[key] = atom
            self._atom_counter += 1
            logger.debug(f"Added node: {atom_type.value} '{name}'")
            return atom
    
    def add_link(self,
                 atom_type: AtomType,
                 outgoing: List[Atom],
                 truth_value: Optional[TruthValue] = None) -> Atom:
        """Add a link to the AtomSpace.
        
        Args:
            atom_type: Type of the link
            outgoing: List of atoms this link connects
            truth_value: Optional truth value
            
        Returns:
            The created or existing link
        """
        with self._lock:
            key = self._make_key(atom_type, "", outgoing)
            if key in self._atoms:
                return self._atoms[key]
            
            link = Atom(
                atom_type=atom_type,
                name="",
                outgoing=outgoing,
                truth_value=truth_value or TruthValue()
            )
            
            # Update incoming sets
            for atom in outgoing:
                atom.incoming.add(link)
            
            self._atoms[key] = link
            self._atom_counter += 1
            logger.debug(f"Added link: {atom_type.value} "
                        f"connecting {len(outgoing)} atoms")
            return link
    
    def get_atom(self, atom_type: AtomType, name: str) -> Optional[Atom]:
        """Get an atom by type and name.
        
        Args:
            atom_type: Type of the atom
            name: Name of the atom
            
        Returns:
            The atom if found, None otherwise
        """
        with self._lock:
            key = self._make_key(atom_type, name, [])
            return self._atoms.get(key)
    
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Get all atoms of a specific type.
        
        Args:
            atom_type: Type of atoms to retrieve
            
        Returns:
            List of atoms matching the type
        """
        with self._lock:
            return [atom for atom in self._atoms.values() 
                   if atom.atom_type == atom_type]
    
    def get_incoming(self, atom: Atom) -> Set[Atom]:
        """Get all atoms that link to this atom.
        
        Args:
            atom: The atom to get incoming links for
            
        Returns:
            Set of atoms that link to this atom
        """
        return atom.incoming
    
    def get_outgoing(self, atom: Atom) -> List[Atom]:
        """Get all atoms this atom links to.
        
        Args:
            atom: The atom to get outgoing links for
            
        Returns:
            List of atoms this atom links to
        """
        return atom.outgoing
    
    def remove_atom(self, atom: Atom) -> bool:
        """Remove an atom from the AtomSpace.
        
        Args:
            atom: The atom to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            key = self._make_key(atom.atom_type, atom.name, atom.outgoing)
            if key not in self._atoms:
                return False
            
            # Remove from incoming sets
            for out_atom in atom.outgoing:
                out_atom.incoming.discard(atom)
            
            # Remove from outgoing incoming sets
            for in_atom in atom.incoming:
                for i, out in enumerate(in_atom.outgoing):
                    if out == atom:
                        in_atom.outgoing[i] = None
            
            del self._atoms[key]
            self._atom_counter -= 1
            logger.debug(f"Removed atom: {atom.atom_type.value}")
            return True
    
    def clear(self):
        """Clear all atoms from the AtomSpace."""
        with self._lock:
            self._atoms.clear()
            self._atom_counter = 0
            logger.info("AtomSpace cleared")
    
    def size(self) -> int:
        """Get the number of atoms in the AtomSpace."""
        return self._atom_counter
    
    def _make_key(self, atom_type: AtomType, name: str, 
                  outgoing: List[Atom]) -> str:
        """Create a unique key for an atom."""
        if outgoing:
            outgoing_keys = [self._make_key(a.atom_type, a.name, a.outgoing) 
                           for a in outgoing]
            return f"{atom_type.value}::{','.join(outgoing_keys)}"
        return f"{atom_type.value}::{name}"
    
    def __str__(self):
        return f"AtomSpace(atoms={self._atom_counter})"
    
    def __repr__(self):
        return self.__str__()
