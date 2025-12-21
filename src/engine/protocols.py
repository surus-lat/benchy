"""Protocol definitions for tasks and interfaces.

These protocols define the contracts that tasks and interfaces must follow.
Tasks are interface-agnostic - they just provide data, prompts, and metrics.
Interfaces handle the specifics of communicating with different AI systems.
"""

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Protocol, Tuple, Any


@dataclass(frozen=True)
class InterfaceCapabilities:
    """Capabilities exposed by an interface implementation."""

    supports_multimodal: bool = False
    supports_logprobs: bool = False
    supports_schema: bool = False
    supports_files: bool = False
    supports_batch: bool = True


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
    def is_multimodal(self) -> bool:
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
        )


def check_compatibility(task: BaseTask, interface: BaseInterface) -> Tuple[bool, str]:
    """Check if a task and interface are compatible.
    
    Args:
        task: Task instance
        interface: Interface instance
        
    Returns:
        Tuple of (is_compatible, reason_if_not)
    """
    # For now, all combinations are compatible
    # Add checks here as needed (e.g., multimodal task + text-only interface)
    
    capabilities = getattr(interface, "capabilities", None)
    supports_multimodal = (
        capabilities.supports_multimodal
        if capabilities is not None
        else getattr(interface, "supports_multimodal", False)
    )
    supports_logprobs = (
        capabilities.supports_logprobs
        if capabilities is not None
        else getattr(interface, "supports_logprobs", False)
    )
    if hasattr(task, 'is_multimodal') and task.is_multimodal:
        if not supports_multimodal:
            return False, "Task requires multimodal support but interface doesn't provide it"

    answer_type = getattr(task, "answer_type", None)
    requires_logprobs = getattr(task, "requires_logprobs", answer_type == "multiple_choice")
    if requires_logprobs:
        if not supports_logprobs:
            return False, "Task requires logprobs for multiple-choice scoring but interface doesn't support logprobs"
    
    return True, ""
