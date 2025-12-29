"""Protocol definitions for tasks and interfaces.

These protocols define the contracts that tasks and interfaces must follow.
Tasks are interface-agnostic - they just provide data, prompts, and metrics.
Interfaces handle the specifics of communicating with different AI systems.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterator, List, Optional, Protocol, Tuple, Any


@dataclass(frozen=True)
class InterfaceCapabilities:
    """Capabilities exposed by an interface implementation."""

    supports_multimodal: bool = False
    supports_logprobs: bool = False
    supports_schema: bool = False
    supports_files: bool = False
    supports_streaming: bool = False
    supports_batch: bool = True
    request_modes: Optional[List[str]] = None


class RequirementLevel(str, Enum):
    """Supported requirement levels for task capabilities."""

    REQUIRED = "required"
    PREFERRED = "preferred"
    OPTIONAL = "optional"


@dataclass(frozen=True)
class TaskCapabilityRequirements:
    """Capability requirements declared by a task."""

    requires_multimodal: RequirementLevel = RequirementLevel.OPTIONAL
    requires_schema: RequirementLevel = RequirementLevel.OPTIONAL
    requires_files: RequirementLevel = RequirementLevel.OPTIONAL
    requires_logprobs: RequirementLevel = RequirementLevel.OPTIONAL
    requires_streaming: RequirementLevel = RequirementLevel.OPTIONAL


@dataclass(frozen=True)
class CompatibilityReport:
    """Compatibility report for a task/interface pair."""

    compatible: bool
    errors: List[str]
    warnings: List[str]
    requirements: TaskCapabilityRequirements
    capabilities: InterfaceCapabilities


def parse_interface_capabilities(
    data: Optional[Dict[str, Any]],
    default: Optional[InterfaceCapabilities] = None,
) -> InterfaceCapabilities:
    """Parse capabilities from config with defaults."""
    default_caps = default or InterfaceCapabilities()
    payload = data or {}
    return InterfaceCapabilities(
        supports_multimodal=payload.get("supports_multimodal", default_caps.supports_multimodal),
        supports_logprobs=payload.get("supports_logprobs", default_caps.supports_logprobs),
        supports_schema=payload.get("supports_schema", default_caps.supports_schema),
        supports_files=payload.get("supports_files", default_caps.supports_files),
        supports_streaming=payload.get("supports_streaming", default_caps.supports_streaming),
        supports_batch=payload.get("supports_batch", default_caps.supports_batch),
        request_modes=payload.get("request_modes", default_caps.request_modes),
    )


def _parse_requirement_level(value: Optional[Any]) -> RequirementLevel:
    if isinstance(value, RequirementLevel):
        return value
    if isinstance(value, str):
        try:
            return RequirementLevel(value.lower())
        except ValueError:
            return RequirementLevel.OPTIONAL
    return RequirementLevel.OPTIONAL



class BaseTask(Protocol):
    """Protocol for benchmark tasks.
    
    Tasks are responsible for:
    - Loading and providing dataset samples
    - Building prompts from samples (for LLM interfaces)
    - Calculating task-specific metrics
    - Declaring capabilities (multimodal, schema requirements, etc.)
    
    Tasks should NOT know about interfaces - they just provide data.
    """
    
    def load(self) -> None:
        """Load the dataset. Called once before evaluation starts."""
        ...
    
    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict]:
        """Iterate over dataset samples.
        
        Each sample should include at minimum:
        - id: Unique identifier for the sample
        - text: Input text (for text-based tasks)
        - expected: Expected output for metrics calculation
        
        Additional fields depend on the task type.
        
        Args:
            limit: Maximum number of samples to return (None for all)
            
        Yields:
            Sample dictionaries
        """
        ...
    
    def get_prompt(self, sample: Dict) -> Tuple[str, str]:
        """Build prompt messages for a sample.
        
        This is used by LLM interfaces to construct chat messages.
        HTTP interfaces may ignore this and use raw sample data.
        
        Args:
            sample: Sample dictionary from get_samples()
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        ...
    
    def get_task_name(self) -> str:
        """Get the task identifier for logging and checkpointing."""
        ...

    @property
    def answer_type(self) -> str:
        """Expected answer type: 'freeform', 'structured', or 'multiple_choice'."""
        ...

    @property
    def requires_logprobs(self) -> bool:
        """Whether this task requires logprobs-based scoring."""
        ...
    
    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate task-specific metrics for a single prediction.
        
        Args:
            prediction: Model output (parsed if applicable)
            expected: Expected output from sample
            sample: Full sample dictionary for additional context
            error: Error message if generation failed (optional)
            error_type: Type of error ('connectivity_error' or 'invalid_response') (optional)
            
        Returns:
            Dictionary of metric names to values. Must include 'valid' (bool) to indicate
            if prediction was usable for metrics calculation.
        """
        ...
    
    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate per-sample metrics into summary statistics.
        
        Args:
            all_metrics: List of per-sample metric dictionaries
            
        Returns:
            Aggregated metrics dictionary
        """
        ...
    
    def get_error_metrics(
        self,
        error: str,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get error metrics structure for failed predictions.
        
        This method is called by the engine when calculate_metrics() raises an exception
        or when a connectivity error occurs. Tasks can override this to provide
        task-specific error metric structures.
        
        Args:
            error: Error message
            error_type: Type of error ('connectivity_error' or 'invalid_response')
            
        Returns:
            Dictionary of error metrics with the same structure as calculate_metrics()
            would return. Must include 'valid': False.
        """
        # Default implementation - tasks can override
        return {
            "valid": False,
            "error": error,
            "error_type": error_type,
            "score": 0.0,
        }
    
    # Optional capability flags - implement as properties
    @property
    def requires_multimodal(self) -> bool:
        """Whether this task requires multimodal inputs (images, audio)."""
        return False
    
    @property
    def requires_schema(self) -> bool:
        """Whether samples include a JSON schema for structured output."""
        return False


class BaseInterface(Protocol):
    """Protocol for AI system interfaces.
    
    Interfaces are responsible for:
    - Preparing requests from task samples
    - Communicating with the AI system
    - Parsing responses to standard format
    
    Interfaces should use task.get_prompt() if they need prompts,
    or access raw sample data for custom endpoints.
    """
    
    def prepare_request(self, sample: Dict, task: BaseTask) -> Dict:
        """Prepare a request from a task sample.
        
        This method adapts task data to the interface's needs:
        - LLM interfaces call task.get_prompt() for chat messages
        - HTTP interfaces may use sample["text"] directly
        
        Args:
            sample: Sample dictionary from task.get_samples()
            task: Task instance for accessing get_prompt() if needed
            
        Returns:
            Request dictionary ready for generate_batch()
        """
        ...
    
    async def generate_batch(self, requests: List[Dict]) -> List[Dict]:
        """Generate outputs for a batch of requests.
        
        Args:
            requests: List of request dicts from prepare_request()
            
        Returns:
            List of result dicts with keys:
            - output: Parsed output (or None on error)
            - raw: Raw response string
            - error: Error message (or None on success)
        """
        ...
    
    async def test_connection(self, max_retries: int = 3, timeout: int = 30) -> bool:
        """Test connection to the AI system.
        
        Args:
            max_retries: Maximum connection attempts
            timeout: Timeout per attempt in seconds
            
        Returns:
            True if connection successful
        """
        ...

    @property
    def supports_multimodal(self) -> bool:
        """Whether this interface supports multimodal inputs."""
        return False

    @property
    def supports_logprobs(self) -> bool:
        """Whether this interface supports logprobs-based scoring."""
        return False

    @property
    def capabilities(self) -> InterfaceCapabilities:
        """Structured capability flags for compatibility checks."""
        return InterfaceCapabilities(
            supports_multimodal=self.supports_multimodal,
            supports_logprobs=self.supports_logprobs,
            supports_schema=False,
            supports_files=False,
            supports_streaming=False,
            supports_batch=True,
        )
def get_task_requirements(task: BaseTask) -> TaskCapabilityRequirements:
    """Resolve task capability requirements with config overrides."""
    requirements = TaskCapabilityRequirements()
    config = getattr(task, "config", {}) or {}
    overrides = getattr(task, "capability_requirements", None)
    if overrides is None:
        overrides = config.get("capability_requirements", {})

    requires_multimodal = getattr(task, "requires_multimodal", False)
    requires_schema = getattr(task, "requires_schema", False)
    answer_type = getattr(task, "answer_type", None)
    requires_logprobs = getattr(task, "requires_logprobs", answer_type == "multiple_choice")

    requirements = TaskCapabilityRequirements(
        requires_multimodal=RequirementLevel.REQUIRED if requires_multimodal else requirements.requires_multimodal,
        requires_schema=RequirementLevel.REQUIRED if requires_schema else requirements.requires_schema,
        requires_files=requirements.requires_files,
        requires_logprobs=RequirementLevel.REQUIRED if requires_logprobs else requirements.requires_logprobs,
        requires_streaming=requirements.requires_streaming,
    )

    if isinstance(overrides, dict):
        requirements = TaskCapabilityRequirements(
            requires_multimodal=_parse_requirement_level(
                overrides.get("requires_multimodal", requirements.requires_multimodal)
            ),
            requires_schema=_parse_requirement_level(
                overrides.get("requires_schema", requirements.requires_schema)
            ),
            requires_files=_parse_requirement_level(
                overrides.get("requires_files", requirements.requires_files)
            ),
            requires_logprobs=_parse_requirement_level(
                overrides.get("requires_logprobs", requirements.requires_logprobs)
            ),
            requires_streaming=_parse_requirement_level(
                overrides.get("requires_streaming", requirements.requires_streaming)
            ),
        )

    return requirements



def check_compatibility(task: BaseTask, interface: BaseInterface) -> CompatibilityReport:
    """Check if a task and interface are compatible."""
    requirements = get_task_requirements(task)
    capabilities = getattr(interface, "capabilities", InterfaceCapabilities())
    errors: List[str] = []
    warnings: List[str] = []

    def evaluate(requirement: RequirementLevel, supported: bool, label: str) -> None:
        if requirement == RequirementLevel.REQUIRED and not supported:
            errors.append(f"Task requires {label} but interface doesn't support it")
        elif requirement == RequirementLevel.PREFERRED and not supported:
            warnings.append(f"Task prefers {label} but interface doesn't support it")

    evaluate(requirements.requires_multimodal, capabilities.supports_multimodal, "multimodal inputs")
    evaluate(requirements.requires_schema, capabilities.supports_schema, "schemas")
    evaluate(requirements.requires_files, capabilities.supports_files, "file inputs")
    evaluate(requirements.requires_logprobs, capabilities.supports_logprobs, "logprobs")
    evaluate(requirements.requires_streaming, capabilities.supports_streaming, "streaming")

    return CompatibilityReport(
        compatible=not errors,
        errors=errors,
        warnings=warnings,
        requirements=requirements,
        capabilities=capabilities,
    )
