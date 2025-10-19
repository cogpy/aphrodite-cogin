# OpenCog Integration with Aphrodite Engine

## Overview

The OpenCog integration brings distributed cognitive architecture capabilities to Aphrodite's large-scale LLM inference engine. This integration enables:

- **Knowledge Representation**: Store and query semantic knowledge using OpenCog's AtomSpace hypergraph database
- **Distributed Cognition**: Process cognitive tasks across multiple workers for scalability
- **Attention Allocation**: Focus computational resources on important information
- **Semantic Grounding**: Ground LLM outputs in a knowledge representation system
- **Cognitive Inference**: Perform reasoning and pattern matching on semantic knowledge

## Architecture

### Components

1. **AtomSpace**: A hypergraph database for storing knowledge as atoms and their relationships
2. **OpenCogEngine**: Main engine that integrates cognitive processing with LLM inference
3. **DistributedCognitionCoordinator**: Coordinates cognitive tasks across multiple workers
4. **OpenCogConfig**: Configuration system for cognitive parameters

### Cognitive Modes

- **Reactive**: Fast, immediate processing with minimal inference
- **Deliberative**: Thorough processing with full inference capabilities
- **Hybrid**: Adaptive processing based on task complexity

### Distribution Strategies

- **Data Parallel**: Distribute tasks across workers
- **Tensor Parallel**: Split atoms across workers
- **Pipeline Parallel**: Process tasks through workers sequentially
- **Hybrid**: Combination of strategies

## Quick Start

### Basic Usage

```python
from aphrodite.opencog import OpenCogEngine, OpenCogConfig

# Create configuration
config = OpenCogConfig(
    enable_opencog=True,
    cognitive_mode="hybrid",
    atomspace_capacity=10000,
)

# Initialize engine
engine = OpenCogEngine(config)

# Ground text in knowledge representation
text = "Artificial intelligence enables cognitive systems."
atoms = engine.ground_text(text)

print(f"Created {len(atoms)} atoms")
print(f"AtomSpace size: {engine.atomspace.size()}")

# Get metrics
metrics = engine.get_metrics()
print(f"Cognitive operations: {metrics['total_cognitive_operations']}")
```

### Distributed Processing

```python
from aphrodite.opencog import OpenCogEngine, OpenCogConfig
from aphrodite.opencog.config import DistributionStrategy

# Configure distributed cognition
config = OpenCogConfig()
config.distributed_config.enable_distributed = True
config.distributed_config.num_cognitive_workers = 4
config.distributed_config.distribution_strategy = DistributionStrategy.HYBRID

# Initialize engine
engine = OpenCogEngine(config)

# Process multiple texts
texts = [
    "Machine learning models process data.",
    "Neural networks recognize patterns.",
    "Deep learning enables AI capabilities.",
]

for text in texts:
    atoms = engine.ground_text(text)
    print(f"Grounded: {text[:40]}... ({len(atoms)} atoms)")

# Synchronize AtomSpaces across workers
engine.synchronize()

# Get distributed statistics
if engine.coordinator:
    stats = engine.coordinator.get_statistics()
    print(f"Workers: {stats['num_workers']}")
    print(f"Total atoms: {stats['total_atoms']}")
```

### Attention Allocation

```python
from aphrodite.opencog import OpenCogEngine, OpenCogConfig
from aphrodite.opencog.atomspace import AtomType

config = OpenCogConfig()
config.attention_config.enable_attention_allocation = True
config.attention_config.max_focus_atoms = 10

engine = OpenCogEngine(config)

# Ground text
atoms = engine.ground_text("Important cognitive information")

# Allocate attention to important atoms
important_atoms = [a for a in atoms if a.atom_type == AtomType.WORD_NODE]
engine.allocate_attention(important_atoms, importance=2.0)

# Get focused atoms
focused = engine.get_focused_atoms(limit=5)
for atom in focused:
    print(f"Focused: {atom.name} (STI: {atom.attention_value.sti:.2f})")

# Decay attention over time
engine.decay_attention()
```

### Knowledge Graph

```python
from aphrodite.opencog import OpenCogEngine, OpenCogConfig
from aphrodite.opencog.atomspace import AtomType

engine = OpenCogEngine(OpenCogConfig())

# Create concepts
ai = engine.atomspace.add_node(AtomType.CONCEPT_NODE, "AI")
ml = engine.atomspace.add_node(AtomType.CONCEPT_NODE, "MachineLearning")
dl = engine.atomspace.add_node(AtomType.CONCEPT_NODE, "DeepLearning")

# Create relationships
ml_inherits_ai = engine.atomspace.add_link(
    AtomType.INHERITANCE_LINK, [ml, ai])
dl_inherits_ml = engine.atomspace.add_link(
    AtomType.INHERITANCE_LINK, [dl, ml])

# Query knowledge graph
concepts = engine.atomspace.get_atoms_by_type(AtomType.CONCEPT_NODE)
print(f"Total concepts: {len(concepts)}")

# Get incoming links (what inherits from AI?)
incoming = engine.atomspace.get_incoming(ai)
print(f"Concepts that inherit from AI: {len(incoming)}")
```

## Configuration

### OpenCogConfig Options

```python
from aphrodite.opencog.config import (
    OpenCogConfig, CognitiveMode, DistributionStrategy
)

config = OpenCogConfig(
    # Core settings
    enable_opencog=True,
    cognitive_mode=CognitiveMode.HYBRID,
    
    # AtomSpace settings
    atomspace_capacity=10000,
    enable_atomspace_persistence=False,
    
    # Integration settings
    integrate_with_sampling=True,
    integrate_with_kv_cache=True,
    enable_semantic_grounding=True,
    
    # Performance settings
    cognitive_batch_size=32,
    max_cognitive_latency_ms=100.0,
    enable_cognitive_caching=True,
)

# Attention configuration
config.attention_config.enable_attention_allocation = True
config.attention_config.sti_threshold = 0.5
config.attention_config.max_focus_atoms = 100

# Inference configuration
config.inference_config.enable_inference = True
config.inference_config.max_inference_steps = 10
config.inference_config.pattern_matching_enabled = True

# Distributed configuration
config.distributed_config.enable_distributed = True
config.distributed_config.num_cognitive_workers = 4
config.distributed_config.distribution_strategy = DistributionStrategy.HYBRID

# Validate configuration
config.validate()
```

## Atom Types

### Node Types

- `CONCEPT_NODE`: General concepts and ideas
- `PREDICATE_NODE`: Predicates and relations
- `WORD_NODE`: Individual words
- `SENTENCE_NODE`: Sentences
- `DOCUMENT_NODE`: Documents
- `VARIABLE_NODE`: Variables for pattern matching
- `SCHEMA_NODE`: Schemas and procedures

### Link Types

- `INHERITANCE_LINK`: Inheritance relationships (is-a)
- `SIMILARITY_LINK`: Similarity relationships
- `MEMBER_LINK`: Set membership
- `LIST_LINK`: Ordered lists
- `EVALUATION_LINK`: Predicate evaluation
- `CONTEXT_LINK`: Contextual relationships
- `INFERENCE_LINK`: Inference relationships

## Integration with Aphrodite Engine

The OpenCog integration can be used alongside Aphrodite's LLM inference:

```python
from aphrodite import LLM, SamplingParams
from aphrodite.opencog import OpenCogEngine, OpenCogConfig

# Initialize LLM
llm = LLM(model="gpt2")

# Initialize OpenCog
opencog_config = OpenCogConfig(enable_opencog=True)
opencog = OpenCogEngine(opencog_config)

# Generate text
prompts = ["Tell me about AI"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)

# Ground LLM output in knowledge representation
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    
    # Ground in AtomSpace
    prompt_atoms = opencog.ground_text(prompt)
    output_atoms = opencog.ground_text(generated_text)
    
    print(f"Prompt atoms: {len(prompt_atoms)}")
    print(f"Output atoms: {len(output_atoms)}")
    print(f"Total AtomSpace size: {opencog.atomspace.size()}")
```

## Performance Considerations

1. **Latency**: Cognitive processing adds latency. Use `max_cognitive_latency_ms` to control
2. **Memory**: AtomSpace consumes memory. Use `atomspace_capacity` to limit
3. **Workers**: More workers increase throughput but use more resources
4. **Caching**: Enable `enable_cognitive_caching` for repeated patterns
5. **Mode**: Use `REACTIVE` mode for lowest latency, `DELIBERATIVE` for best quality

## Examples

See `examples/opencog_example.py` for comprehensive examples including:

- Basic text grounding
- Distributed processing
- Attention allocation
- Knowledge graph creation and querying

Run the examples:

```bash
python examples/opencog_example.py
```

## Testing

Run the OpenCog tests:

```bash
# Standalone tests (no dependencies)
python tests/opencog/test_standalone.py

# Unit tests (requires pytest)
pytest tests/opencog/
```

## API Reference

### AtomSpace

```python
class AtomSpace:
    def add_node(atom_type: AtomType, name: str, 
                truth_value: Optional[TruthValue] = None) -> Atom
    def add_link(atom_type: AtomType, outgoing: List[Atom],
                truth_value: Optional[TruthValue] = None) -> Atom
    def get_atom(atom_type: AtomType, name: str) -> Optional[Atom]
    def get_atoms_by_type(atom_type: AtomType) -> List[Atom]
    def get_incoming(atom: Atom) -> Set[Atom]
    def get_outgoing(atom: Atom) -> List[Atom]
    def remove_atom(atom: Atom) -> bool
    def clear() -> None
    def size() -> int
```

### OpenCogEngine

```python
class OpenCogEngine:
    def __init__(config: Optional[OpenCogConfig] = None)
    def ground_text(text: str, context: Optional[Dict] = None) -> List[Atom]
    def allocate_attention(atoms: List[Atom], importance: float = 1.0) -> None
    def decay_attention() -> None
    def get_focused_atoms(limit: Optional[int] = None) -> List[Atom]
    def get_metrics() -> Dict[str, Any]
    def synchronize() -> None
    def reset() -> None
```

## Future Enhancements

- Integration with pattern mining
- Reinforcement learning for attention allocation
- Probabilistic logic networks (PLN) reasoning
- OpenCog Hyperon integration
- Cross-worker attention synchronization
- Persistent AtomSpace storage
- Real-time cognitive metrics dashboard

## License

This integration follows the same license as the Aphrodite Engine project.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## References

- [OpenCog Project](https://opencog.org/)
- [Aphrodite Engine Documentation](https://aphrodite.pygmalion.chat)
- [AtomSpace Documentation](https://wiki.opencog.org/w/AtomSpace)
