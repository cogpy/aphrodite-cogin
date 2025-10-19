"""Standalone test for OpenCog components.

This test file can run independently without requiring torch or other
heavy dependencies.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly without going through aphrodite.__init__
from aphrodite.opencog.atomspace import (
    AtomSpace, AtomType, TruthValue, AttentionValue, Atom
)
from aphrodite.opencog.config import (
    OpenCogConfig, CognitiveMode, DistributionStrategy
)


def test_atomspace():
    """Test basic AtomSpace functionality."""
    print("Testing AtomSpace...")
    
    atomspace = AtomSpace()
    assert atomspace.size() == 0, "New atomspace should be empty"
    
    # Add nodes
    node1 = atomspace.add_node(AtomType.CONCEPT_NODE, "AI")
    node2 = atomspace.add_node(AtomType.CONCEPT_NODE, "ML")
    
    assert atomspace.size() == 2, "Should have 2 nodes"
    
    # Add link
    link = atomspace.add_link(
        AtomType.INHERITANCE_LINK, [node2, node1])
    
    assert atomspace.size() == 3, "Should have 2 nodes + 1 link"
    assert len(link.outgoing) == 2, "Link should have 2 outgoing atoms"
    
    # Test incoming
    incoming = atomspace.get_incoming(node1)
    assert len(incoming) == 1, "Node1 should have 1 incoming link"
    
    print("  ✓ AtomSpace basic operations work")
    
    # Test truth values
    tv = TruthValue(strength=0.8, confidence=0.9)
    node3 = atomspace.add_node(AtomType.CONCEPT_NODE, "DL", tv)
    
    assert node3.truth_value.strength == 0.8
    assert node3.truth_value.confidence == 0.9
    
    print("  ✓ Truth values work")
    
    # Test attention
    av = AttentionValue(sti=1.0, lti=0.5)
    node1.attention_value = av
    
    assert node1.attention_value.sti == 1.0
    
    print("  ✓ Attention values work")
    
    # Test retrieval
    retrieved = atomspace.get_atom(AtomType.CONCEPT_NODE, "AI")
    assert retrieved == node1
    
    concepts = atomspace.get_atoms_by_type(AtomType.CONCEPT_NODE)
    assert len(concepts) == 3
    
    print("  ✓ Atom retrieval works")
    
    print("✅ All AtomSpace tests passed!\n")


def test_config():
    """Test OpenCog configuration."""
    print("Testing OpenCog Config...")
    
    config = OpenCogConfig()
    assert config.enable_opencog == True
    assert config.cognitive_mode == CognitiveMode.HYBRID
    
    print("  ✓ Default config created")
    
    # Validate
    config.validate()  # Should not raise
    
    print("  ✓ Config validation works")
    
    # Test to_dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert "enable_opencog" in config_dict
    
    print("  ✓ Config serialization works")
    
    # Test from_dict
    config2 = OpenCogConfig.from_dict(config_dict)
    assert config2.enable_opencog == config.enable_opencog
    
    print("  ✓ Config deserialization works")
    
    print("✅ All Config tests passed!\n")


def test_integration():
    """Test integration of components."""
    print("Testing component integration...")
    
    # Import engine directly
    from aphrodite.opencog.engine import OpenCogEngine
    
    config = OpenCogConfig(
        enable_opencog=True,
        cognitive_mode=CognitiveMode.REACTIVE
    )
    config.distributed_config.enable_distributed = False
    
    engine = OpenCogEngine(config)
    
    assert engine.atomspace.size() == 0
    
    print("  ✓ Engine created")
    
    # Ground text
    atoms = engine.ground_text("Hello world.")
    assert len(atoms) > 0
    assert engine.atomspace.size() > 0
    
    print(f"  ✓ Text grounding works ({len(atoms)} atoms created)")
    
    # Get metrics
    metrics = engine.get_metrics()
    assert metrics["total_cognitive_operations"] > 0
    assert metrics["atomspace_size"] > 0
    
    print("  ✓ Metrics collection works")
    
    # Test attention
    word_atoms = [a for a in atoms if a.atom_type == AtomType.WORD_NODE]
    if word_atoms:
        engine.allocate_attention(word_atoms[:1], importance=1.0)
        focused = engine.get_focused_atoms(limit=5)
        assert len(focused) > 0
        print("  ✓ Attention allocation works")
    
    # Reset
    engine.reset()
    assert engine.atomspace.size() == 0
    
    print("  ✓ Engine reset works")
    
    print("✅ All integration tests passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("OpenCog Standalone Tests")
    print("=" * 60 + "\n")
    
    try:
        test_atomspace()
        test_config()
        test_integration()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
