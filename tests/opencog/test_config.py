"""Tests for OpenCog configuration."""

import unittest
from aphrodite.opencog.config import (
    OpenCogConfig, CognitiveMode, DistributionStrategy,
    AttentionConfig, InferenceConfig, DistributedConfig
)


class TestOpenCogConfig(unittest.TestCase):
    """Test cases for OpenCog configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = OpenCogConfig()
        
        self.assertTrue(config.enable_opencog)
        self.assertEqual(config.cognitive_mode, CognitiveMode.HYBRID)
        self.assertEqual(config.atomspace_capacity, 10000)
    
    def test_validate_valid_config(self):
        """Test validation with valid config."""
        config = OpenCogConfig()
        
        # Should not raise any exception
        config.validate()
    
    def test_validate_invalid_capacity(self):
        """Test validation with invalid capacity."""
        config = OpenCogConfig(atomspace_capacity=-1)
        
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_validate_invalid_thresholds(self):
        """Test validation with invalid thresholds."""
        config = OpenCogConfig()
        config.attention_config.sti_threshold = 1.5
        
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "enable_opencog": True,
            "cognitive_mode": "reactive",
            "atomspace_capacity": 5000,
        }
        
        config = OpenCogConfig.from_dict(config_dict)
        
        self.assertTrue(config.enable_opencog)
        self.assertEqual(config.cognitive_mode, CognitiveMode.REACTIVE)
        self.assertEqual(config.atomspace_capacity, 5000)
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = OpenCogConfig()
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertIn("enable_opencog", config_dict)
        self.assertIn("cognitive_mode", config_dict)
        self.assertEqual(config_dict["cognitive_mode"], "hybrid")
    
    def test_attention_config(self):
        """Test attention configuration."""
        config = AttentionConfig(
            enable_attention_allocation=True,
            sti_threshold=0.6,
            max_focus_atoms=50
        )
        
        self.assertTrue(config.enable_attention_allocation)
        self.assertEqual(config.sti_threshold, 0.6)
        self.assertEqual(config.max_focus_atoms, 50)
    
    def test_inference_config(self):
        """Test inference configuration."""
        config = InferenceConfig(
            enable_inference=True,
            max_inference_steps=5
        )
        
        self.assertTrue(config.enable_inference)
        self.assertEqual(config.max_inference_steps, 5)
    
    def test_distributed_config(self):
        """Test distributed configuration."""
        config = DistributedConfig(
            enable_distributed=True,
            distribution_strategy=DistributionStrategy.TENSOR_PARALLEL,
            num_cognitive_workers=4
        )
        
        self.assertTrue(config.enable_distributed)
        self.assertEqual(config.distribution_strategy,
                        DistributionStrategy.TENSOR_PARALLEL)
        self.assertEqual(config.num_cognitive_workers, 4)


if __name__ == "__main__":
    unittest.main()
