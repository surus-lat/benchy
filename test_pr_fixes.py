#!/usr/bin/env python3
"""Test suite for PR review fixes.

Tests the following fixes:
1. MSE metric returns NaN instead of 0.0 on parse failure
2. MultimodalStructuredHandler validates schema when requires_schema=True
3. MultipleChoiceHandler has List import for type annotations
"""

import math
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# Test 1: MSE metric returns NaN on parse failure
def test_mse_parse_failure():
    """Test that MSE returns NaN when conversion fails, not 0.0."""
    from src.tasks.common.metrics import MeanSquaredError
    
    mse = MeanSquaredError()
    
    # Valid inputs should work
    result1 = mse.compute("5.0", "3.0", {})
    assert result1 == 4.0, f"Expected 4.0, got {result1}"
    
    # Invalid prediction should return NaN
    result2 = mse.compute("not_a_number", "3.0", {})
    assert math.isnan(result2), f"Expected NaN on invalid prediction, got {result2}"
    
    # Invalid expected should return NaN
    result3 = mse.compute("5.0", "not_a_number", {})
    assert math.isnan(result3), f"Expected NaN on invalid expected, got {result3}"
    
    # Both invalid should return NaN
    result4 = mse.compute("invalid", "invalid", {})
    assert math.isnan(result4), f"Expected NaN on both invalid, got {result4}"
    
    # Test aggregation excludes NaN values
    values = [
        {"mse": 1.0},
        {"mse": 4.0},
        {"mse": float('nan')},  # Should be excluded
        {"mse": 9.0},
    ]
    aggregated = mse.aggregate(values)
    assert aggregated["mse"] == (1.0 + 4.0 + 9.0) / 3, f"Expected mean of 4.67, got {aggregated['mse']}"
    
    # Test that all NaN values result in 0.0 mean
    values_all_nan = [
        {"mse": float('nan')},
        {"mse": float('nan')},
    ]
    aggregated_nan = mse.aggregate(values_all_nan)
    assert aggregated_nan["mse"] == 0.0, f"Expected 0.0 when all NaN, got {aggregated_nan['mse']}"
    
    print("✓ Test 1 passed: MSE returns NaN on parse failure and aggregates correctly")


# Test 2: Schema validation in MultimodalStructuredHandler
def test_schema_validation():
    """Test that MultimodalStructuredHandler raises error when schema missing and required."""
    from src.tasks.common.multimodal_structured import MultimodalStructuredHandler
    
    # Create a temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        source_dir = tmp_path / "source"
        data_dir = tmp_path / "data"
        source_dir.mkdir()
        data_dir.mkdir()
        
        # Create a handler that requires schema
        class TestHandler(MultimodalStructuredHandler):
            requires_schema = True
        
        handler = TestHandler(config={"source_dir": str(source_dir)})
        handler.data_dir = data_dir
        handler.source_path = source_dir
        
        # Create expected.json but no schema.json
        expected_file = data_dir / "expected.json"
        expected_file.write_text(json.dumps({"test": {"value": 1}}))
        
        # Should raise FileNotFoundError when schema is missing
        try:
            handler.load()
            assert False, "Expected FileNotFoundError when schema missing and required"
        except FileNotFoundError as e:
            assert "schema" in str(e).lower(), f"Error message should mention schema: {e}"
            assert "requires_schema" in str(e).lower(), f"Error message should mention requires_schema: {e}"
            print("✓ Test 2a passed: Raises FileNotFoundError when schema missing and required")
        
        # Test that it works when schema exists
        schema_file = data_dir / "schema.json"
        schema_file.write_text(json.dumps({"type": "object", "properties": {"test": {"type": "string"}}}))
        
        handler2 = TestHandler(config={"source_dir": str(source_dir)})
        handler2.data_dir = data_dir
        handler2.source_path = source_dir
        
        # Should not raise error when schema exists
        try:
            handler2.load()
            assert handler2.schema is not None, "Schema should be loaded"
            print("✓ Test 2b passed: Loads successfully when schema exists")
        except Exception as e:
            assert False, f"Should not raise error when schema exists: {e}"
        
        # Test that handler without requires_schema doesn't raise error
        class TestHandlerNoSchema(MultimodalStructuredHandler):
            requires_schema = False
        
        handler3 = TestHandlerNoSchema(config={"source_dir": str(source_dir)})
        handler3.data_dir = data_dir
        handler3.source_path = source_dir
        
        # Remove schema to test
        schema_file.unlink()
        
        try:
            handler3.load()
            assert handler3.schema is None, "Schema should be None when not required and missing"
            print("✓ Test 2c passed: Does not raise error when requires_schema=False")
        except Exception as e:
            assert False, f"Should not raise error when requires_schema=False: {e}"


# Test 3: List import in MultipleChoiceHandler
def test_list_import():
    """Test that List is imported and aggregate_metrics type annotation works."""
    from src.tasks.common.multiple_choice import MultipleChoiceHandler
    
    # Just importing should not raise NameError
    try:
        handler = MultipleChoiceHandler()
        print("✓ Test 3a passed: MultipleChoiceHandler imports without NameError")
    except NameError as e:
        assert False, f"NameError on import: {e}"
    
    # Test that aggregate_metrics can be called (type annotation should work)
    test_metrics = [
        {"valid": True, "correct": True, "accuracy": 1.0},
        {"valid": True, "correct": False, "accuracy": 0.0},
        {"valid": False, "correct": False, "accuracy": 0.0},
    ]
    
    try:
        result = handler.aggregate_metrics(test_metrics)
        assert "accuracy" in result, "Result should contain accuracy"
        assert result["accuracy"] == 0.5, f"Expected 0.5 accuracy, got {result['accuracy']}"
        print("✓ Test 3b passed: aggregate_metrics works correctly with List type annotation")
    except Exception as e:
        assert False, f"aggregate_metrics should work: {e}"


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Running PR Fix Tests")
    print("=" * 60)
    
    tests = [
        ("MSE Parse Failure", test_mse_parse_failure),
        ("Schema Validation", test_schema_validation),
        ("List Import", test_list_import),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

