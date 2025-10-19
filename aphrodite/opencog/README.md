# OpenCog Integration Module

This module provides OpenCog cognitive architecture integration for Aphrodite Engine, enabling distributed cognition over large-scale LLM inference.

## Features

- **AtomSpace**: Hypergraph knowledge representation
- **Distributed Processing**: Multi-worker cognitive processing
- **Attention Mechanism**: Dynamic focus allocation
- **Semantic Grounding**: Ground LLM outputs in knowledge graphs
- **Cognitive Inference**: Pattern matching and reasoning

## Quick Start

```python
from aphrodite.opencog import OpenCogEngine, OpenCogConfig

# Initialize
config = OpenCogConfig(enable_opencog=True)
engine = OpenCogEngine(config)

# Ground text
atoms = engine.ground_text("AI enables cognitive systems.")
print(f"Created {len(atoms)} atoms")

# Get metrics
print(engine.get_metrics())
```

## Documentation

See `/docs/OPENCOG.md` for comprehensive documentation.

## Examples

Run the examples:

```bash
python examples/opencog_example.py
```

## Testing

```bash
python tests/opencog/test_standalone.py
```

## Modules

- `atomspace.py`: Hypergraph knowledge representation
- `config.py`: Configuration system
- `engine.py`: Main cognitive engine
- `distributed.py`: Distributed processing coordinator
