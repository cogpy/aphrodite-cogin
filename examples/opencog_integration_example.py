"""Example of integrating OpenCog with Aphrodite Engine.

This example shows how to use OpenCog cognitive processing
alongside Aphrodite LLM inference.
"""

# This example demonstrates the integration pattern.
# In a real scenario, this would be integrated into the main engine.

import logging
logging.basicConfig(level=logging.INFO)

print("=" * 60)
print("OpenCog + Aphrodite Integration Example")
print("=" * 60)

# Mock some Aphrodite types for demonstration
class MockLLMOutput:
    def __init__(self, text):
        self.text = text

def mock_llm_generate(prompt):
    """Mock LLM generation."""
    # In reality, this would use the actual Aphrodite engine
    responses = {
        "What is AI?": "Artificial Intelligence is the simulation of human intelligence by machines.",
        "Explain machine learning": "Machine learning is a subset of AI that learns from data.",
    }
    return MockLLMOutput(responses.get(prompt, "I don't know."))

# Load integration modules
import sys
sys.path.insert(0, '.')

import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

print("\nLoading OpenCog modules...")
integration_mod = load_module('integration', 'aphrodite/opencog/integration.py')

# Load dependencies
atomspace_mod = load_module('atomspace', 'aphrodite/opencog/atomspace.py')
config_mod = load_module('config', 'aphrodite/opencog/config.py')
distributed_mod = load_module('distributed', 'aphrodite/opencog/distributed.py')
engine_mod = load_module('engine', 'aphrodite/opencog/engine.py')

# Wire up dependencies
distributed_mod.AtomSpace = atomspace_mod.AtomSpace
distributed_mod.Atom = atomspace_mod.Atom
distributed_mod.AtomType = atomspace_mod.AtomType
distributed_mod.OpenCogConfig = config_mod.OpenCogConfig
distributed_mod.DistributionStrategy = config_mod.DistributionStrategy

engine_mod.AtomSpace = atomspace_mod.AtomSpace
engine_mod.Atom = atomspace_mod.Atom
engine_mod.AtomType = atomspace_mod.AtomType
engine_mod.TruthValue = atomspace_mod.TruthValue
engine_mod.OpenCogConfig = config_mod.OpenCogConfig
engine_mod.CognitiveMode = config_mod.CognitiveMode
engine_mod.DistributedCognitionCoordinator = distributed_mod.DistributedCognitionCoordinator
engine_mod.CognitiveTask = distributed_mod.CognitiveTask
engine_mod.CognitiveResult = distributed_mod.CognitiveResult

# Mock types
class MockSequenceGroup:
    def __init__(self, request_id, prompt):
        self.request_id = request_id
        self.prompt = prompt

engine_mod.SequenceGroup = MockSequenceGroup
engine_mod.SequenceGroupOutput = None
engine_mod.RequestOutput = None

integration_mod.OpenCogEngine = engine_mod.OpenCogEngine
integration_mod.OpenCogConfig = config_mod.OpenCogConfig

create_opencog_integration = integration_mod.create_opencog_integration

print("✓ Modules loaded\n")

# Create OpenCog integration
print("1. Initializing OpenCog integration...")
opencog = create_opencog_integration(
    enable=True,
    config_dict={
        "enable_opencog": True,
        "cognitive_mode": "reactive",
        "atomspace_capacity": 5000,
        "distributed_config": {
            "enable_distributed": False
        }
    }
)

if opencog.enabled:
    print("   ✓ OpenCog integration enabled")
else:
    print("   ✗ OpenCog integration disabled")
    exit(1)

# Simulate LLM inference with OpenCog processing
print("\n2. Processing prompts with cognitive grounding...")
prompts = [
    "What is AI?",
    "Explain machine learning",
]

for i, prompt in enumerate(prompts, 1):
    print(f"\n   Prompt {i}: {prompt}")
    
    # Process prompt through OpenCog
    cog_result = opencog.process_prompt(prompt, request_id=f"req_{i}")
    if cog_result:
        print(f"   ✓ Cognitive processing:")
        print(f"      - Atoms created: {cog_result['atoms_created']}")
        print(f"      - Focused atoms: {cog_result['focused_atoms']}")
        print(f"      - AtomSpace size: {cog_result['atomspace_size']}")
    
    # Generate response (mock)
    response = mock_llm_generate(prompt)
    print(f"   ✓ LLM Response: {response.text[:60]}...")
    
    # Process output through OpenCog
    output_result = opencog.process_output(response.text, request_id=f"req_{i}")
    if output_result:
        print(f"   ✓ Output grounding:")
        print(f"      - Atoms created: {output_result['atoms_created']}")
        print(f"      - Total AtomSpace size: {output_result['atomspace_size']}")

# Get metrics
print("\n3. Cognitive metrics...")
metrics = opencog.get_metrics()
if metrics:
    print(f"   ✓ Total cognitive operations: {metrics['total_cognitive_operations']}")
    print(f"   ✓ Total atoms created: {metrics['total_atoms_created']}")
    print(f"   ✓ Total processing time: {metrics['total_processing_time_ms']:.2f}ms")
    print(f"   ✓ Final AtomSpace size: {metrics['atomspace_size']}")

# Summary
print("\n" + "=" * 60)
print("✅ Integration example completed!")
print("=" * 60)
print("\nKey integration points:")
print("  • OpenCogIntegration class wraps the engine")
print("  • process_prompt() grounds prompts in AtomSpace")
print("  • process_output() grounds LLM outputs")
print("  • get_metrics() tracks cognitive processing")
print("  • Minimal overhead, optional enablement")
print("\nTo integrate with real Aphrodite engine:")
print("  1. Add OpenCogIntegration to engine initialization")
print("  2. Call process_prompt() before LLM inference")
print("  3. Call process_output() after generation")
print("  4. Use metrics for monitoring")
