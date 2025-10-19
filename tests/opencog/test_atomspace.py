"""Tests for OpenCog AtomSpace implementation."""

import unittest
from aphrodite.opencog.atomspace import (
    AtomSpace, AtomType, Atom, TruthValue, AttentionValue
)


class TestAtomSpace(unittest.TestCase):
    """Test cases for AtomSpace."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.atomspace = AtomSpace()
    
    def tearDown(self):
        """Clean up after tests."""
        self.atomspace.clear()
    
    def test_initialization(self):
        """Test AtomSpace initialization."""
        self.assertIsNotNone(self.atomspace)
        self.assertEqual(self.atomspace.size(), 0)
    
    def test_add_node(self):
        """Test adding nodes to AtomSpace."""
        node = self.atomspace.add_node(
            AtomType.CONCEPT_NODE, "test_concept")
        
        self.assertIsNotNone(node)
        self.assertEqual(node.atom_type, AtomType.CONCEPT_NODE)
        self.assertEqual(node.name, "test_concept")
        self.assertEqual(self.atomspace.size(), 1)
    
    def test_add_duplicate_node(self):
        """Test that duplicate nodes return the same atom."""
        node1 = self.atomspace.add_node(
            AtomType.CONCEPT_NODE, "duplicate")
        node2 = self.atomspace.add_node(
            AtomType.CONCEPT_NODE, "duplicate")
        
        self.assertEqual(node1, node2)
        self.assertEqual(self.atomspace.size(), 1)
    
    def test_add_link(self):
        """Test adding links to AtomSpace."""
        node1 = self.atomspace.add_node(
            AtomType.CONCEPT_NODE, "parent")
        node2 = self.atomspace.add_node(
            AtomType.CONCEPT_NODE, "child")
        
        link = self.atomspace.add_link(
            AtomType.INHERITANCE_LINK, [node2, node1])
        
        self.assertIsNotNone(link)
        self.assertEqual(link.atom_type, AtomType.INHERITANCE_LINK)
        self.assertEqual(len(link.outgoing), 2)
        self.assertEqual(self.atomspace.size(), 3)
    
    def test_get_atom(self):
        """Test retrieving atoms by type and name."""
        self.atomspace.add_node(AtomType.CONCEPT_NODE, "test")
        
        retrieved = self.atomspace.get_atom(
            AtomType.CONCEPT_NODE, "test")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "test")
    
    def test_get_atoms_by_type(self):
        """Test retrieving all atoms of a specific type."""
        self.atomspace.add_node(AtomType.CONCEPT_NODE, "concept1")
        self.atomspace.add_node(AtomType.CONCEPT_NODE, "concept2")
        self.atomspace.add_node(AtomType.WORD_NODE, "word1")
        
        concepts = self.atomspace.get_atoms_by_type(
            AtomType.CONCEPT_NODE)
        
        self.assertEqual(len(concepts), 2)
        for concept in concepts:
            self.assertEqual(concept.atom_type, AtomType.CONCEPT_NODE)
    
    def test_incoming_links(self):
        """Test incoming link tracking."""
        node1 = self.atomspace.add_node(
            AtomType.CONCEPT_NODE, "target")
        node2 = self.atomspace.add_node(
            AtomType.CONCEPT_NODE, "source")
        
        link = self.atomspace.add_link(
            AtomType.SIMILARITY_LINK, [node2, node1])
        
        incoming = self.atomspace.get_incoming(node1)
        
        self.assertEqual(len(incoming), 1)
        self.assertIn(link, incoming)
    
    def test_outgoing_links(self):
        """Test outgoing link retrieval."""
        node1 = self.atomspace.add_node(
            AtomType.CONCEPT_NODE, "node1")
        node2 = self.atomspace.add_node(
            AtomType.CONCEPT_NODE, "node2")
        
        link = self.atomspace.add_link(
            AtomType.LIST_LINK, [node1, node2])
        
        outgoing = self.atomspace.get_outgoing(link)
        
        self.assertEqual(len(outgoing), 2)
        self.assertIn(node1, outgoing)
        self.assertIn(node2, outgoing)
    
    def test_remove_atom(self):
        """Test removing atoms from AtomSpace."""
        node = self.atomspace.add_node(
            AtomType.CONCEPT_NODE, "to_remove")
        
        self.assertEqual(self.atomspace.size(), 1)
        
        result = self.atomspace.remove_atom(node)
        
        self.assertTrue(result)
        self.assertEqual(self.atomspace.size(), 0)
    
    def test_clear(self):
        """Test clearing all atoms."""
        self.atomspace.add_node(AtomType.CONCEPT_NODE, "node1")
        self.atomspace.add_node(AtomType.CONCEPT_NODE, "node2")
        self.atomspace.add_node(AtomType.WORD_NODE, "word1")
        
        self.assertEqual(self.atomspace.size(), 3)
        
        self.atomspace.clear()
        
        self.assertEqual(self.atomspace.size(), 0)
    
    def test_truth_value(self):
        """Test truth value assignment."""
        tv = TruthValue(strength=0.8, confidence=0.9)
        node = self.atomspace.add_node(
            AtomType.CONCEPT_NODE, "test_tv", tv)
        
        self.assertEqual(node.truth_value.strength, 0.8)
        self.assertEqual(node.truth_value.confidence, 0.9)
    
    def test_truth_value_bounds(self):
        """Test that truth values are bounded."""
        tv = TruthValue(strength=1.5, confidence=-0.5)
        
        self.assertEqual(tv.strength, 1.0)
        self.assertEqual(tv.confidence, 0.0)
    
    def test_attention_value(self):
        """Test attention value creation."""
        av = AttentionValue(sti=1.0, lti=0.5, vlti=0.1)
        
        self.assertEqual(av.sti, 1.0)
        self.assertEqual(av.lti, 0.5)
        self.assertEqual(av.vlti, 0.1)
    
    def test_atom_metadata(self):
        """Test atom metadata storage."""
        node = self.atomspace.add_node(
            AtomType.CONCEPT_NODE, "meta_test")
        
        node.metadata["key1"] = "value1"
        node.metadata["key2"] = 42
        
        self.assertEqual(node.metadata["key1"], "value1")
        self.assertEqual(node.metadata["key2"], 42)


if __name__ == "__main__":
    unittest.main()
