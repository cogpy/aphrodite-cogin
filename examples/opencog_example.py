"""Example demonstrating OpenCog integration with Aphrodite Engine.

This example shows how to use the OpenCog cognitive architecture
for distributed cognition over large-scale LLM inference.
"""

from aphrodite.opencog import OpenCogEngine, OpenCogConfig, AtomType
from aphrodite.opencog.config import CognitiveMode, DistributionStrategy


def basic_example():
    """Basic example of OpenCog integration."""
    print("=" * 60)
    print("Basic OpenCog Example")
    print("=" * 60)
    
    # Create configuration
    config = OpenCogConfig(
        enable_opencog=True,
        cognitive_mode=CognitiveMode.HYBRID,
        atomspace_capacity=1000,
    )
    
    # Initialize engine
    engine = OpenCogEngine(config)
    print(f"\nInitialized: {engine}")
    
    # Ground some text
    text = "The quick brown fox jumps over the lazy dog."
    print(f"\nGrounding text: '{text}'")
    atoms = engine.ground_text(text)
    print(f"Created {len(atoms)} atoms")
    
    # Show some atoms
    print("\nSample atoms created:")
    for i, atom in enumerate(atoms[:5]):
        print(f"  {i+1}. {atom.atom_type.value}: {atom.name or '<link>'}")
    
    # Get metrics
    metrics = engine.get_metrics()
    print(f"\nMetrics: {metrics}")
    
    print("\n" + "=" * 60)


def distributed_example():
    """Example with distributed cognition."""
    print("=" * 60)
    print("Distributed Cognition Example")
    print("=" * 60)
    
    # Create configuration with distributed processing
    config = OpenCogConfig(
        enable_opencog=True,
        cognitive_mode=CognitiveMode.DELIBERATIVE,
        atomspace_capacity=5000,
    )
    
    # Enable distributed cognition
    config.distributed_config.enable_distributed = True
    config.distributed_config.num_cognitive_workers = 4
    config.distributed_config.distribution_strategy = \
        DistributionStrategy.HYBRID
    
    # Initialize engine
    engine = OpenCogEngine(config)
    print(f"\nInitialized distributed engine: {engine}")
    
    # Ground multiple texts
    texts = [
        "Artificial intelligence is transforming technology.",
        "Machine learning models process vast amounts of data.",
        "Neural networks can recognize complex patterns.",
        "Deep learning enables advanced cognitive capabilities.",
    ]
    
    print(f"\nGrounding {len(texts)} texts...")
    total_atoms = 0
    for i, text in enumerate(texts):
        atoms = engine.ground_text(text)
        total_atoms += len(atoms)
        print(f"  {i+1}. '{text[:40]}...' -> {len(atoms)} atoms")
    
    print(f"\nTotal atoms created: {total_atoms}")
    
    # Synchronize atomspaces
    if engine.coordinator:
        print("\nSynchronizing distributed AtomSpaces...")
        engine.synchronize()
        
        # Get distributed statistics
        stats = engine.coordinator.get_statistics()
        print(f"\nDistributed statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Get final metrics
    metrics = engine.get_metrics()
    print(f"\nFinal metrics:")
    for key, value in metrics.items():
        if key != "distributed":
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)


def attention_example():
    """Example demonstrating attention allocation."""
    print("=" * 60)
    print("Attention Allocation Example")
    print("=" * 60)
    
    # Create configuration with attention enabled
    config = OpenCogConfig(
        enable_opencog=True,
        cognitive_mode=CognitiveMode.REACTIVE,
    )
    config.attention_config.enable_attention_allocation = True
    config.attention_config.max_focus_atoms = 5
    
    # Initialize engine
    engine = OpenCogEngine(config)
    print(f"\nInitialized: {engine}")
    
    # Ground text
    text = "Cognitive attention focuses on important information."
    print(f"\nGrounding text: '{text}'")
    atoms = engine.ground_text(text)
    
    # Allocate attention to some atoms
    important_atoms = [a for a in atoms if a.atom_type == AtomType.WORD_NODE][:3]
    print(f"\nAllocating attention to {len(important_atoms)} word atoms...")
    engine.allocate_attention(important_atoms, importance=2.0)
    
    # Get focused atoms
    focused = engine.get_focused_atoms(limit=5)
    print(f"\nAtoms in attentional focus:")
    for i, atom in enumerate(focused):
        print(f"  {i+1}. {atom.atom_type.value}: {atom.name or '<link>'} "
              f"(STI: {atom.attention_value.sti:.2f})")
    
    # Decay attention
    print("\nDecaying attention...")
    engine.decay_attention()
    
    # Check focused atoms again
    focused = engine.get_focused_atoms(limit=5)
    print(f"\nAtoms in attentional focus after decay:")
    for i, atom in enumerate(focused):
        print(f"  {i+1}. {atom.atom_type.value}: {atom.name or '<link>'} "
              f"(STI: {atom.attention_value.sti:.2f})")
    
    print("\n" + "=" * 60)


def knowledge_graph_example():
    """Example showing knowledge graph creation."""
    print("=" * 60)
    print("Knowledge Graph Example")
    print("=" * 60)
    
    config = OpenCogConfig(enable_opencog=True)
    engine = OpenCogEngine(config)
    print(f"\nInitialized: {engine}")
    
    # Create a simple knowledge graph
    print("\nBuilding knowledge graph...")
    
    # Create concept nodes
    ai = engine.atomspace.add_node(AtomType.CONCEPT_NODE, "AI")
    ml = engine.atomspace.add_node(AtomType.CONCEPT_NODE, "MachineLearning")
    dl = engine.atomspace.add_node(AtomType.CONCEPT_NODE, "DeepLearning")
    
    # Create inheritance relationships
    ml_inherits_ai = engine.atomspace.add_link(
        AtomType.INHERITANCE_LINK, [ml, ai])
    dl_inherits_ml = engine.atomspace.add_link(
        AtomType.INHERITANCE_LINK, [dl, ml])
    
    print("  - AI concept")
    print("  - Machine Learning concept (inherits from AI)")
    print("  - Deep Learning concept (inherits from Machine Learning)")
    
    # Query the knowledge graph
    print("\nQuerying knowledge graph:")
    
    # Get all concepts
    concepts = engine.atomspace.get_atoms_by_type(AtomType.CONCEPT_NODE)
    print(f"\nTotal concepts: {len(concepts)}")
    for concept in concepts:
        print(f"  - {concept.name}")
    
    # Get relationships
    relationships = engine.atomspace.get_atoms_by_type(
        AtomType.INHERITANCE_LINK)
    print(f"\nInheritance relationships: {len(relationships)}")
    for rel in relationships:
        if len(rel.outgoing) >= 2:
            child = rel.outgoing[0].name
            parent = rel.outgoing[1].name
            print(f"  - {child} inherits from {parent}")
    
    # Get incoming links for AI
    incoming = engine.atomspace.get_incoming(ai)
    print(f"\nConcepts that inherit from AI: {len(incoming)}")
    for link in incoming:
        if link.outgoing:
            print(f"  - {link.outgoing[0].name}")
    
    print("\n" + "=" * 60)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("OpenCog + Aphrodite Engine Examples")
    print("=" * 60 + "\n")
    
    try:
        # Run examples
        basic_example()
        print("\n")
        
        distributed_example()
        print("\n")
        
        attention_example()
        print("\n")
        
        knowledge_graph_example()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
