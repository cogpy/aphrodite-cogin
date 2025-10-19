"""Tests for OpenCog Engine."""

import unittest
from aphrodite.opencog.engine import OpenCogEngine
from aphrodite.opencog.config import OpenCogConfig, CognitiveMode
from aphrodite.opencog.atomspace import AtomType


class TestOpenCogEngine(unittest.TestCase):
    """Test cases for OpenCog Engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = OpenCogConfig(
            enable_opencog=True,
            cognitive_mode=CognitiveMode.REACTIVE,
        )
        config.distributed_config.enable_distributed = False
        self.engine = OpenCogEngine(config)
    
    def tearDown(self):
        """Clean up after tests."""
        self.engine.reset()
    
    def test_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.atomspace)
        self.assertEqual(self.engine.atomspace.size(), 0)
    
    def test_ground_text(self):
        """Test text grounding."""
        text = "Hello world."
        atoms = self.engine.ground_text(text)
        
        self.assertIsNotNone(atoms)
        self.assertGreater(len(atoms), 0)
        self.assertGreater(self.engine.atomspace.size(), 0)
    
    def test_ground_text_with_context(self):
        """Test text grounding with context."""
        text = "Test sentence."
        context = {"user": "test_user", "session": "123"}
        atoms = self.engine.ground_text(text, context)
        
        self.assertIsNotNone(atoms)
        self.assertGreater(len(atoms), 0)
    
    def test_allocate_attention(self):
        """Test attention allocation."""
        atoms = self.engine.ground_text("Test text.")
        word_atoms = [a for a in atoms if a.atom_type == AtomType.WORD_NODE]
        
        if word_atoms:
            initial_sti = word_atoms[0].attention_value.sti
            self.engine.allocate_attention(word_atoms, importance=1.0)
            final_sti = word_atoms[0].attention_value.sti
            
            self.assertGreater(final_sti, initial_sti)
    
    def test_decay_attention(self):
        """Test attention decay."""
        atoms = self.engine.ground_text("Test text.")
        word_atoms = [a for a in atoms if a.atom_type == AtomType.WORD_NODE]
        
        if word_atoms:
            self.engine.allocate_attention(word_atoms, importance=2.0)
            initial_sti = word_atoms[0].attention_value.sti
            
            self.engine.decay_attention()
            final_sti = word_atoms[0].attention_value.sti
            
            self.assertLess(final_sti, initial_sti)
    
    def test_get_focused_atoms(self):
        """Test getting focused atoms."""
        self.engine.ground_text("Test sentence for attention.")
        focused = self.engine.get_focused_atoms(limit=5)
        
        self.assertIsInstance(focused, list)
        self.assertLessEqual(len(focused), 5)
    
    def test_get_metrics(self):
        """Test metrics retrieval."""
        self.engine.ground_text("Test text.")
        metrics = self.engine.get_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_cognitive_operations", metrics)
        self.assertIn("total_atoms_created", metrics)
        self.assertIn("atomspace_size", metrics)
        self.assertGreater(metrics["total_cognitive_operations"], 0)
    
    def test_reset(self):
        """Test engine reset."""
        self.engine.ground_text("Test text.")
        self.assertGreater(self.engine.atomspace.size(), 0)
        
        self.engine.reset()
        
        self.assertEqual(self.engine.atomspace.size(), 0)
        metrics = self.engine.get_metrics()
        self.assertEqual(metrics["total_cognitive_operations"], 0)


class TestOpenCogEngineDistributed(unittest.TestCase):
    """Test cases for distributed OpenCog Engine."""
    
    def test_distributed_initialization(self):
        """Test distributed engine initialization."""
        config = OpenCogConfig(cognitive_mode=CognitiveMode.DELIBERATIVE)
        config.distributed_config.enable_distributed = True
        config.distributed_config.num_cognitive_workers = 2
        
        engine = OpenCogEngine(config)
        
        self.assertIsNotNone(engine.coordinator)
        self.assertEqual(engine.coordinator.num_workers, 2)


if __name__ == "__main__":
    unittest.main()
