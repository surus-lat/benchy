"""Spanish task metrics calculator.

Calculates accuracy for multiple choice tasks and exact_match for generate_until tasks.
"""

import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SpanishMetricsCalculator:
    """Calculator for Spanish task metrics: accuracy and exact_match."""
    
    def __init__(self, config: Dict):
        """Initialize the metrics calculator.
        
        Args:
            config: Configuration dictionary (may contain metric settings)
        """
        self.config = config
    
    def _parse_choice_from_response(self, response: str, choices: List[str]) -> Optional[int]:
        """Parse the selected choice from model response.
        
        Tries multiple strategies:
        1. Look for choice letter (A, B, C, etc.)
        2. Look for full choice text match
        3. Look for partial match
        
        Args:
            response: Model response text
            choices: List of choice strings
            
        Returns:
            Index of selected choice (0-based), or None if not found
        """
        response_lower = response.strip().lower()
        
        # Strategy 1: Look for choice letter at start or after common prefixes
        choice_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        for i, letter in enumerate(choice_letters[:len(choices)]):
            # Match letter at start, or after "opción", "respuesta", etc.
            pattern = rf'\b{letter}\b'
            if re.search(pattern, response_lower, re.IGNORECASE):
                return i
        
        # Strategy 2: Look for full choice text match
        for i, choice in enumerate(choices):
            choice_lower = choice.lower().strip()
            if choice_lower in response_lower:
                return i
        
        # Strategy 3: Look for partial match (first 10 chars)
        for i, choice in enumerate(choices):
            choice_lower = choice.lower().strip()
            if len(choice_lower) > 10 and choice_lower[:10] in response_lower:
                return i
        
        return None
    
    def _extract_number_from_response(self, response: str) -> Optional[str]:
        """Extract numeric answer from response (for generate_until tasks like mgsm).
        
        Args:
            response: Model response text
            
        Returns:
            Extracted number as string, or None if not found
        """
        def normalize_number(raw: str) -> Optional[str]:
            raw = raw.strip().replace(" ", "")
            if not raw:
                return None
            sign = ""
            if raw.startswith(("+", "-")):
                sign = "-" if raw.startswith("-") else ""
                raw = raw[1:]
            if not raw:
                return None
            if "." in raw and "," in raw:
                digits = re.sub(r"[.,]", "", raw)
                return f"{sign}{digits}" if digits else None
            if "." in raw or "," in raw:
                sep = "." if "." in raw else ","
                parts = raw.split(sep)
                if all(part.isdigit() for part in parts) and len(parts[-1]) == 3:
                    digits = "".join(parts)
                    return f"{sign}{digits}" if digits else None
                try:
                    value = float(raw.replace(",", "."))
                    if value.is_integer():
                        return f"{sign}{int(value)}"
                    normalized = f"{value}".rstrip("0").rstrip(".")
                    return f"{sign}{normalized}"
                except ValueError:
                    pass
            digits = re.sub(r"\D", "", raw)
            return f"{sign}{digits}" if digits else None

        number_pattern = r"[-+]?\d[\d.,]*"
        response_tail = re.split(r"(?i)respuesta|answer", response)[-1]
        numbers = re.findall(number_pattern, response_tail)
        if not numbers:
            numbers = re.findall(number_pattern, response)
        if not numbers:
            return None
        return normalize_number(numbers[-1])
    
    def calculate(
        self,
        prediction: Optional[Any],
        expected: Any,
        sample: Dict,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        task_type: str = "multiple_choice",
    ) -> Dict[str, Any]:
        """Calculate metrics for a single prediction.
        
        Args:
            prediction: Model output (text response)
            expected: Expected answer (choice index for multiple choice, number for generate_until)
            sample: Full sample dict (contains choices for multiple choice)
            error: Error message if generation failed (optional)
            error_type: Type of error ('connectivity_error' or 'invalid_response') (optional)
            task_type: Type of task ('multiple_choice' or 'generate_until')
            
        Returns:
            Dictionary with metrics: {"acc": float, "valid": bool} or {"exact_match": float, "valid": bool}
        """
        # Handle errors
        if error or prediction is None:
            if task_type == "multiple_choice":
                return {
                    "valid": False,
                    "error": error or "No prediction",
                    "error_type": error_type,
                    "acc": 0.0,
                }
            else:  # generate_until
                return {
                    "valid": False,
                    "error": error or "No prediction",
                    "error_type": error_type,
                    "exact_match": 0.0,
                }
        
        if task_type == "multiple_choice":
            # Handle logprobs: prediction is already an integer index
            if isinstance(prediction, int):
                choices = sample.get("choices", [])
                if not choices:
                    return {
                        "valid": False,
                        "error": "No choices in sample",
                        "error_type": "invalid_response",
                        "acc": 0.0,
                    }
                
                selected_idx = prediction
                if selected_idx < 0 or selected_idx >= len(choices):
                    return {
                        "valid": False,
                        "error": f"Invalid choice index: {selected_idx} (max: {len(choices)-1})",
                        "error_type": "invalid_response",
                        "acc": 0.0,
                    }
            else:
                # Parse choice from text response
                prediction_text = str(prediction).strip() if prediction else ""
                
                if not prediction_text:
                    return {
                        "valid": False,
                        "error": "Empty prediction",
                        "error_type": "invalid_response",
                        "acc": 0.0,
                    }
                
                choices = sample.get("choices", [])
                if not choices:
                    return {
                        "valid": False,
                        "error": "No choices in sample",
                        "error_type": "invalid_response",
                        "acc": 0.0,
                    }
                
                selected_idx = self._parse_choice_from_response(prediction_text, choices)
                
                if selected_idx is None:
                    return {
                        "valid": False,
                        "error": f"Could not parse choice from response: {prediction_text[:100]}",
                        "error_type": "invalid_response",
                        "acc": 0.0,
                    }
            
            # Compare to expected answer
            # Expected can be an index (int) or a label (str that matches a choice)
            if isinstance(expected, int):
                expected_idx = expected
            elif isinstance(expected, str):
                # Try to find expected string in choices
                try:
                    expected_idx = choices.index(expected)
                except ValueError:
                    # Try case-insensitive match
                    expected_lower = expected.lower()
                    expected_idx = next(
                        (i for i, c in enumerate(choices) if c.lower() == expected_lower),
                        None
                    )
                    if expected_idx is None:
                        return {
                            "valid": False,
                            "error": f"Expected answer '{expected}' not found in choices",
                            "error_type": "invalid_response",
                            "acc": 0.0,
                        }
            else:
                return {
                    "valid": False,
                    "error": f"Unexpected expected type: {type(expected)}",
                    "error_type": "invalid_response",
                    "acc": 0.0,
                }
            
            is_correct = (selected_idx == expected_idx)
            
            return {
                "valid": True,
                "acc": 1.0 if is_correct else 0.0,
                "selected_idx": selected_idx,
                "expected_idx": expected_idx,
            }
        
        else:  # generate_until
            # Extract number from response
            prediction_text = str(prediction).strip() if prediction else ""
            
            if not prediction_text:
                return {
                    "valid": False,
                    "error": "Empty prediction",
                    "error_type": "invalid_response",
                    "exact_match": 0.0,
                }
            
            extracted_number = self._extract_number_from_response(prediction_text)
            
            if extracted_number is None:
                return {
                    "valid": False,
                    "error": f"Could not extract number from response: {prediction_text[:100]}",
                    "error_type": "invalid_response",
                    "exact_match": 0.0,
                }
            
            # Compare to expected (should be a number string)
            expected_str = str(expected).strip()
            is_match = (extracted_number == expected_str)
            
            return {
                "valid": True,
                "exact_match": 1.0 if is_match else 0.0,
                "extracted_number": extracted_number,
                "expected_number": expected_str,
            }
    
    def aggregate(self, all_metrics: List[Dict], weight_by_size: bool = True) -> Dict[str, Any]:
        """Aggregate per-sample metrics into summary statistics.
        
        Args:
            all_metrics: List of per-sample metric dictionaries
            weight_by_size: If True, weight accuracy by sample size (for aggregation across subtasks)
            
        Returns:
            Aggregated metrics dictionary
        """
        if not all_metrics:
            return {
                "total_samples": 0,
                "valid_samples": 0,
                "acc": 0.0,
                "exact_match": 0.0,
            }
        
        valid_metrics = [m for m in all_metrics if m.get("valid", False)]
        total_samples = len(all_metrics)
        valid_samples = len(valid_metrics)
        
        if valid_samples == 0:
            return {
                "total_samples": total_samples,
                "valid_samples": 0,
                "acc": 0.0,
                "exact_match": 0.0,
                "error_rate": 1.0,
            }
        
        # Calculate accuracy (for multiple choice tasks)
        acc_metrics = [m for m in valid_metrics if "acc" in m]
        if acc_metrics:
            acc_scores = [float(m.get("acc", 0.0)) for m in acc_metrics]
            acc = sum(acc_scores) / len(acc_scores) if acc_scores else 0.0
        else:
            acc = None
        
        # Calculate exact_match (for generate_until tasks)
        exact_match_metrics = [m for m in valid_metrics if "exact_match" in m]
        if exact_match_metrics:
            exact_match_scores = [float(m.get("exact_match", 0.0)) for m in exact_match_metrics]
            exact_match = sum(exact_match_scores) / len(exact_match_scores) if exact_match_scores else 0.0
        else:
            exact_match = None
        
        aggregated = {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "error_rate": (total_samples - valid_samples) / total_samples if total_samples > 0 else 0.0,
        }
        
        if acc is not None:
            aggregated["acc"] = acc
        if exact_match is not None:
            aggregated["exact_match"] = exact_match
        
        return aggregated
