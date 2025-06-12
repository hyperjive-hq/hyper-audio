"""Pipeline configuration management and validation."""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from .stage_interface import EnhancedPipelineStage


@dataclass
class StageConfig:
    """Configuration for a single pipeline stage."""
    stage_class: str          # e.g., "VoiceSeparator"
    stage_id: str             # Unique instance ID
    config: Dict[str, Any]    # Stage-specific configuration
    enabled: bool = True


@dataclass
class ConnectionConfig:
    """Configuration for connecting stages."""
    from_stage: str           # Source stage ID
    from_output: str          # Output name from source stage
    to_stage: str             # Target stage ID
    to_input: str             # Input name for target stage


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    name: str
    description: str
    stages: List[StageConfig]
    connections: List[ConnectionConfig]
    global_config: Dict[str, Any]


class ConfigManager:
    """Manages pipeline configuration loading, validation, and saving."""

    def __init__(self):
        self.registered_stages: Dict[str, type] = {}

    def register_stage(self, stage_class: type, name: str = None):
        """Register a stage class for use in configurations.
        
        Args:
            stage_class: Class implementing EnhancedPipelineStage
            name: Optional name (defaults to class name)
        """
        if not issubclass(stage_class, EnhancedPipelineStage):
            raise ValueError(f"Stage class {stage_class} must inherit from EnhancedPipelineStage")

        stage_name = name or stage_class.__name__
        self.registered_stages[stage_name] = stage_class

    def load_config(self, config_path: Path) -> PipelineConfig:
        """Load pipeline configuration from file.
        
        Args:
            config_path: Path to YAML or JSON config file
            
        Returns:
            Parsed pipeline configuration
        """
        config_path = Path(config_path)

        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                raw_config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                raw_config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

        return self._parse_config(raw_config)

    def save_config(self, config: PipelineConfig, config_path: Path):
        """Save pipeline configuration to file.
        
        Args:
            config: Pipeline configuration
            config_path: Output file path
        """
        config_path = Path(config_path)
        config_dict = asdict(config)

        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(config_dict, f, indent=2, sort_keys=False)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

    def _parse_config(self, raw_config: Dict[str, Any]) -> PipelineConfig:
        """Parse raw config dictionary into PipelineConfig."""
        stages = []
        for stage_data in raw_config.get('stages', []):
            stages.append(StageConfig(
                stage_class=stage_data['stage_class'],
                stage_id=stage_data['stage_id'],
                config=stage_data.get('config', {}),
                enabled=stage_data.get('enabled', True)
            ))

        connections = []
        for conn_data in raw_config.get('connections', []):
            connections.append(ConnectionConfig(
                from_stage=conn_data['from_stage'],
                from_output=conn_data['from_output'],
                to_stage=conn_data['to_stage'],
                to_input=conn_data['to_input']
            ))

        return PipelineConfig(
            name=raw_config.get('name', 'Unnamed Pipeline'),
            description=raw_config.get('description', ''),
            stages=stages,
            connections=connections,
            global_config=raw_config.get('global_config', {})
        )

    def validate_config(self, config: PipelineConfig) -> Tuple[bool, List[str]]:
        """Validate pipeline configuration.
        
        Args:
            config: Pipeline configuration to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        stage_instances = {}

        # 1. Check that all stage classes are registered
        for stage_config in config.stages:
            if not stage_config.enabled:
                continue

            if stage_config.stage_class not in self.registered_stages:
                errors.append(f"Unknown stage class: {stage_config.stage_class}")
                continue

            # Create stage instance for validation
            try:
                stage_class = self.registered_stages[stage_config.stage_class]
                stage_instance = stage_class(
                    stage_id=stage_config.stage_id,
                    config=stage_config.config
                )
                stage_instances[stage_config.stage_id] = stage_instance
            except Exception as e:
                errors.append(f"Failed to create stage {stage_config.stage_id}: {e}")

        # 2. Check that stage IDs are unique
        stage_ids = [s.stage_id for s in config.stages if s.enabled]
        if len(stage_ids) != len(set(stage_ids)):
            errors.append("Stage IDs must be unique")

        # 3. Validate connections
        for conn in config.connections:
            # Check that referenced stages exist
            if conn.from_stage not in stage_instances:
                errors.append(f"Connection references unknown source stage: {conn.from_stage}")
                continue
            if conn.to_stage not in stage_instances:
                errors.append(f"Connection references unknown target stage: {conn.to_stage}")
                continue

            from_stage = stage_instances[conn.from_stage]
            to_stage = stage_instances[conn.to_stage]

            # Check that output exists
            if conn.from_output not in from_stage.get_output_names():
                errors.append(
                    f"Stage {conn.from_stage} does not have output '{conn.from_output}'. "
                    f"Available: {from_stage.get_output_names()}"
                )
                continue

            # Check that input exists
            if conn.to_input not in [inp.name for inp in to_stage._metadata.inputs]:
                errors.append(
                    f"Stage {conn.to_stage} does not have input '{conn.to_input}'. "
                    f"Available: {[inp.name for inp in to_stage._metadata.inputs]}"
                )
                continue

            # Check data type compatibility
            from_output = next(out for out in from_stage._metadata.outputs if out.name == conn.from_output)
            to_input = next(inp for inp in to_stage._metadata.inputs if inp.name == conn.to_input)

            if from_output.data_type != to_input.data_type:
                errors.append(
                    f"Type mismatch in connection {conn.from_stage}.{conn.from_output} → "
                    f"{conn.to_stage}.{conn.to_input}: {from_output.data_type.value} → {to_input.data_type.value}"
                )

        # 4. Check that all required inputs are satisfied
        for stage_id, stage in stage_instances.items():
            required_inputs = stage.get_required_inputs()
            connected_inputs = [conn.to_input for conn in config.connections if conn.to_stage == stage_id]

            missing_inputs = set(required_inputs) - set(connected_inputs)
            if missing_inputs:
                errors.append(f"Stage {stage_id} has unsatisfied required inputs: {missing_inputs}")

        # 5. Check for circular dependencies
        if self._has_circular_dependencies(config):
            errors.append("Pipeline has circular dependencies")

        return len(errors) == 0, errors

    def _has_circular_dependencies(self, config: PipelineConfig) -> bool:
        """Check if the pipeline has circular dependencies."""
        # Build dependency graph
        graph = {}
        for stage in config.stages:
            if stage.enabled:
                graph[stage.stage_id] = []

        for conn in config.connections:
            if conn.from_stage in graph and conn.to_stage in graph:
                graph[conn.from_stage].append(conn.to_stage)

        # DFS to detect cycles
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True

        return False

    def get_execution_order(self, config: PipelineConfig) -> List[str]:
        """Get the order in which stages should be executed.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            List of stage IDs in execution order
        """
        # Build dependency graph
        graph = {}
        in_degree = {}

        for stage in config.stages:
            if stage.enabled:
                stage_id = stage.stage_id
                graph[stage_id] = []
                in_degree[stage_id] = 0

        for conn in config.connections:
            if conn.from_stage in graph and conn.to_stage in graph:
                graph[conn.from_stage].append(conn.to_stage)
                in_degree[conn.to_stage] += 1

        # Topological sort using Kahn's algorithm
        queue = [stage_id for stage_id, degree in in_degree.items() if degree == 0]
        execution_order = []

        while queue:
            current = queue.pop(0)
            execution_order.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return execution_order

    def create_example_config(self) -> PipelineConfig:
        """Create an example pipeline configuration."""
        return PipelineConfig(
            name="Speech Enhancement Pipeline",
            description="Multi-stage speech enhancement with music separation and noise reduction",
            stages=[
                StageConfig(
                    stage_class="AudioPreprocessor",
                    stage_id="preprocessor",
                    config={"normalize": True, "target_sr": 44100}
                ),
                StageConfig(
                    stage_class="VoiceSeparator",
                    stage_id="music_separator",
                    config={"model_name": "htdemucs_ft"}
                ),
                StageConfig(
                    stage_class="SpeechEnhancer",
                    stage_id="noise_reducer",
                    config={"model_name": "speechbrain/sepformer-whamr"}
                ),
                StageConfig(
                    stage_class="SpeakerDiarizer",
                    stage_id="diarizer",
                    config={"min_speakers": 1, "max_speakers": 10}
                )
            ],
            connections=[
                ConnectionConfig("preprocessor", "audio_with_sr", "music_separator", "audio_input"),
                ConnectionConfig("music_separator", "vocals", "noise_reducer", "audio_input"),
                ConnectionConfig("noise_reducer", "enhanced_audio", "diarizer", "audio_input")
            ],
            global_config={
                "device": "cuda",
                "cache_dir": "~/.cache/hyper_audio",
                "checkpoint_interval": 5
            }
        )
