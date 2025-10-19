"""OpenCog configuration for Aphrodite Engine integration."""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class CognitiveMode(Enum):
    """Cognitive processing modes."""
    REACTIVE = "reactive"  # Fast, reactive processing
    DELIBERATIVE = "deliberative"  # Slower, more thoughtful processing
    HYBRID = "hybrid"  # Combination of both


class DistributionStrategy(Enum):
    """Strategies for distributing cognitive processing."""
    TENSOR_PARALLEL = "tensor_parallel"  # Distribute across tensor parallel workers
    PIPELINE_PARALLEL = "pipeline_parallel"  # Pipeline-based distribution
    DATA_PARALLEL = "data_parallel"  # Data-parallel distribution
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class AttentionConfig:
    """Configuration for attention allocation."""
    enable_attention_allocation: bool = True
    sti_threshold: float = 0.5  # Short-term importance threshold
    lti_threshold: float = 0.3  # Long-term importance threshold
    max_focus_atoms: int = 100  # Maximum atoms in attentional focus
    attention_decay_rate: float = 0.95  # Rate of attention decay


@dataclass
class InferenceConfig:
    """Configuration for cognitive inference."""
    enable_inference: bool = True
    max_inference_steps: int = 10
    inference_confidence_threshold: float = 0.7
    pattern_matching_enabled: bool = True
    backward_chaining_enabled: bool = True
    forward_chaining_enabled: bool = True


@dataclass
class DistributedConfig:
    """Configuration for distributed cognition."""
    enable_distributed: bool = True
    distribution_strategy: DistributionStrategy = DistributionStrategy.HYBRID
    num_cognitive_workers: Optional[int] = None  # Auto-detect if None
    sync_interval_ms: int = 100  # Synchronization interval
    enable_knowledge_sharing: bool = True
    shard_atomspace: bool = True  # Shard AtomSpace across workers


@dataclass
class OpenCogConfig:
    """Main configuration for OpenCog integration with Aphrodite.
    
    This configuration defines how OpenCog's cognitive architecture
    integrates with Aphrodite's LLM inference engine for distributed
    cognition at scale.
    """
    
    # Core settings
    enable_opencog: bool = True
    cognitive_mode: CognitiveMode = CognitiveMode.HYBRID
    
    # AtomSpace settings
    atomspace_capacity: int = 10000  # Maximum atoms in memory
    enable_atomspace_persistence: bool = False
    persistence_path: Optional[str] = None
    
    # Attention allocation
    attention_config: AttentionConfig = field(default_factory=AttentionConfig)
    
    # Cognitive inference
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Distributed cognition
    distributed_config: DistributedConfig = field(
        default_factory=DistributedConfig)
    
    # Integration settings
    integrate_with_sampling: bool = True  # Integrate with LLM sampling
    integrate_with_kv_cache: bool = True  # Use KV cache for cognitive memory
    enable_semantic_grounding: bool = True  # Ground LLM outputs in AtomSpace
    
    # Performance settings
    cognitive_batch_size: int = 32  # Batch size for cognitive processing
    max_cognitive_latency_ms: float = 100.0  # Max added latency
    enable_cognitive_caching: bool = True
    
    # Logging and monitoring
    log_cognitive_operations: bool = False
    track_cognitive_metrics: bool = True
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if self.atomspace_capacity <= 0:
            raise ValueError("atomspace_capacity must be positive")
        
        if self.attention_config.sti_threshold < 0 or \
           self.attention_config.sti_threshold > 1:
            raise ValueError("sti_threshold must be in [0, 1]")
        
        if self.attention_config.lti_threshold < 0 or \
           self.attention_config.lti_threshold > 1:
            raise ValueError("lti_threshold must be in [0, 1]")
        
        if self.inference_config.max_inference_steps <= 0:
            raise ValueError("max_inference_steps must be positive")
        
        if self.cognitive_batch_size <= 0:
            raise ValueError("cognitive_batch_size must be positive")
        
        if self.max_cognitive_latency_ms <= 0:
            raise ValueError("max_cognitive_latency_ms must be positive")
        
        if self.enable_atomspace_persistence and not self.persistence_path:
            raise ValueError(
                "persistence_path required when persistence is enabled")
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "OpenCogConfig":
        """Create configuration from dictionary."""
        # Convert nested dicts to dataclass instances
        if "attention_config" in config_dict:
            config_dict["attention_config"] = AttentionConfig(
                **config_dict["attention_config"])
        
        if "inference_config" in config_dict:
            config_dict["inference_config"] = InferenceConfig(
                **config_dict["inference_config"])
        
        if "distributed_config" in config_dict:
            config_dict["distributed_config"] = DistributedConfig(
                **config_dict["distributed_config"])
        
        # Convert enum strings to enums
        if "cognitive_mode" in config_dict:
            config_dict["cognitive_mode"] = CognitiveMode(
                config_dict["cognitive_mode"])
        
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "enable_opencog": self.enable_opencog,
            "cognitive_mode": self.cognitive_mode.value,
            "atomspace_capacity": self.atomspace_capacity,
            "enable_atomspace_persistence": self.enable_atomspace_persistence,
            "persistence_path": self.persistence_path,
            "attention_config": {
                "enable_attention_allocation": 
                    self.attention_config.enable_attention_allocation,
                "sti_threshold": self.attention_config.sti_threshold,
                "lti_threshold": self.attention_config.lti_threshold,
                "max_focus_atoms": self.attention_config.max_focus_atoms,
                "attention_decay_rate": 
                    self.attention_config.attention_decay_rate,
            },
            "inference_config": {
                "enable_inference": self.inference_config.enable_inference,
                "max_inference_steps": 
                    self.inference_config.max_inference_steps,
                "inference_confidence_threshold": 
                    self.inference_config.inference_confidence_threshold,
                "pattern_matching_enabled": 
                    self.inference_config.pattern_matching_enabled,
                "backward_chaining_enabled": 
                    self.inference_config.backward_chaining_enabled,
                "forward_chaining_enabled": 
                    self.inference_config.forward_chaining_enabled,
            },
            "distributed_config": {
                "enable_distributed": 
                    self.distributed_config.enable_distributed,
                "distribution_strategy": 
                    self.distributed_config.distribution_strategy.value,
                "num_cognitive_workers": 
                    self.distributed_config.num_cognitive_workers,
                "sync_interval_ms": 
                    self.distributed_config.sync_interval_ms,
                "enable_knowledge_sharing": 
                    self.distributed_config.enable_knowledge_sharing,
                "shard_atomspace": self.distributed_config.shard_atomspace,
            },
            "integrate_with_sampling": self.integrate_with_sampling,
            "integrate_with_kv_cache": self.integrate_with_kv_cache,
            "enable_semantic_grounding": self.enable_semantic_grounding,
            "cognitive_batch_size": self.cognitive_batch_size,
            "max_cognitive_latency_ms": self.max_cognitive_latency_ms,
            "enable_cognitive_caching": self.enable_cognitive_caching,
            "log_cognitive_operations": self.log_cognitive_operations,
            "track_cognitive_metrics": self.track_cognitive_metrics,
        }
