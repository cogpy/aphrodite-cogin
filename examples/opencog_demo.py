#!/usr/bin/env python3
"""Simple demonstration of OpenCog integration.

This script demonstrates the OpenCog integration without requiring
the full Aphrodite dependency chain.
"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct imports to avoid dependency issues
import importlib.util

def load_module(name, path):
    """Load a module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load OpenCog modules
print("Loading OpenCog modules...")
atomspace_mod = load_module('atomspace', 'aphrodite/opencog/atomspace.py')
config_mod = load_module('config', 'aphrodite/opencog/config.py')

# Get main classes
AtomSpace = atomspace_mod.AtomSpace
AtomType = atomspace_mod.AtomType
TruthValue = atomspace_mod.TruthValue
OpenCogConfig = config_mod.OpenCogConfig
CognitiveMode = config_mod.CognitiveMode

print("✓ Modules loaded\n")

# Demonstration
print("=" * 60)
print("OpenCog + Aphrodite Integration Demo")
print("=" * 60)

# 1. Create AtomSpace
print("\n1. Creating AtomSpace...")
atomspace = AtomSpace()
print(f"   ✓ AtomSpace created: {atomspace}")

# 2. Add concepts
print("\n2. Building knowledge graph...")
ai = atomspace.add_node(AtomType.CONCEPT_NODE, "ArtificialIntelligence")
ml = atomspace.add_node(AtomType.CONCEPT_NODE, "MachineLearning")
dl = atomspace.add_node(AtomType.CONCEPT_NODE, "DeepLearning")
nlp = atomspace.add_node(AtomType.CONCEPT_NODE, "NaturalLanguageProcessing")

print(f"   ✓ Added 4 concept nodes")

# 3. Add relationships
ml_inherits_ai = atomspace.add_link(
    AtomType.INHERITANCE_LINK, [ml, ai],
    TruthValue(strength=0.9, confidence=0.8)
)
dl_inherits_ml = atomspace.add_link(
    AtomType.INHERITANCE_LINK, [dl, ml],
    TruthValue(strength=0.95, confidence=0.9)
)
nlp_inherits_ai = atomspace.add_link(
    AtomType.INHERITANCE_LINK, [nlp, ai],
    TruthValue(strength=0.85, confidence=0.8)
)

print(f"   ✓ Added 3 inheritance links")
print(f"   ✓ Total atoms in graph: {atomspace.size()}")

# 4. Query knowledge graph
print("\n3. Querying knowledge graph...")
concepts = atomspace.get_atoms_by_type(AtomType.CONCEPT_NODE)
print(f"   ✓ Found {len(concepts)} concepts:")
for i, concept in enumerate(concepts, 1):
    print(f"      {i}. {concept.name}")

# 5. Find inheritance relationships
print("\n4. Finding relationships...")
inheritances = atomspace.get_atoms_by_type(AtomType.INHERITANCE_LINK)
print(f"   ✓ Found {len(inheritances)} inheritance relationships:")
for i, link in enumerate(inheritances, 1):
    if len(link.outgoing) >= 2:
        child = link.outgoing[0].name
        parent = link.outgoing[1].name
        strength = link.truth_value.strength
        print(f"      {i}. {child} → {parent} (strength: {strength:.2f})")

# 6. Find what inherits from AI
print("\n5. What inherits from ArtificialIntelligence?")
incoming = atomspace.get_incoming(ai)
print(f"   ✓ Found {len(incoming)} direct descendants:")
for link in incoming:
    if link.outgoing:
        child = link.outgoing[0].name
        print(f"      - {child}")

# 7. Configuration
print("\n6. Testing configuration...")
config = OpenCogConfig(
    enable_opencog=True,
    cognitive_mode=CognitiveMode.HYBRID,
    atomspace_capacity=10000
)
print(f"   ✓ Config created: mode={config.cognitive_mode.value}")
print(f"   ✓ AtomSpace capacity: {config.atomspace_capacity}")
print(f"   ✓ Distributed enabled: {config.distributed_config.enable_distributed}")

# Validate config
try:
    config.validate()
    print(f"   ✓ Configuration validated successfully")
except Exception as e:
    print(f"   ✗ Configuration validation failed: {e}")

# 8. Simulate text grounding
print("\n7. Simulating text grounding...")
text = "AI and machine learning transform technology"
words = text.lower().split()

for word in words:
    word_node = atomspace.add_node(
        AtomType.WORD_NODE, word,
        TruthValue(strength=1.0, confidence=0.9)
    )

print(f"   ✓ Grounded text into {len(words)} word nodes")
print(f"   ✓ Final AtomSpace size: {atomspace.size()}")

# 9. Summary statistics
print("\n8. Summary Statistics...")
all_atoms = []
for atom_type in AtomType:
    atoms = atomspace.get_atoms_by_type(atom_type)
    if atoms:
        all_atoms.extend(atoms)
        print(f"   {atom_type.value}: {len(atoms)}")

print(f"\n   Total atoms: {atomspace.size()}")

print("\n" + "=" * 60)
print("✅ Demo completed successfully!")
print("=" * 60)
print("\nKey capabilities demonstrated:")
print("  • Hypergraph knowledge representation")
print("  • Concept nodes and relationships")
print("  • Truth values for uncertain knowledge")
print("  • Graph querying and traversal")
print("  • Configuration management")
print("  • Text grounding")
print("\nFor full integration with Aphrodite LLM engine,")
print("see docs/OPENCOG.md")
