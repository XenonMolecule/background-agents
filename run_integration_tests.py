#!/usr/bin/env python3
"""
Simple test runner for integration tests without pytest dependency.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up test environment
os.environ["PRECURSOR_DISABLE_CALENDAR"] = "1"

def run_test(test_func, test_name):
    """Run a single test function with proper setup."""
    print(f"Running {test_name}...")
    
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            db_path = tmp_path / "test_scratchpad.db"
            
            # Set up environment
            old_db_path = os.environ.get("PRECURSOR_SCRATCHPAD_DB")
            os.environ["PRECURSOR_SCRATCHPAD_DB"] = str(db_path)
            
            try:
                # Create a simple monkeypatch-like object
                class MockMonkeypatch:
                    def __init__(self):
                        self._patches = []
                    
                    def setattr(self, obj, name, value):
                        old_value = getattr(obj, name, None)
                        setattr(obj, name, value)
                        self._patches.append((obj, name, old_value))
                    
                    def setenv(self, name, value):
                        old_value = os.environ.get(name)
                        os.environ[name] = value
                        self._patches.append(("env", name, old_value))
                    
                    def cleanup(self):
                        for patch in reversed(self._patches):
                            if patch[0] == "env":
                                if patch[2] is None:
                                    os.environ.pop(patch[1], None)
                                else:
                                    os.environ[patch[1]] = patch[2]
                            else:
                                if patch[2] is None:
                                    delattr(patch[0], patch[1])
                                else:
                                    setattr(patch[0], patch[1], patch[2])
                
                monkeypatch = MockMonkeypatch()
                
                # Run the test
                test_func(tmp_path, monkeypatch)
                
                # Cleanup
                monkeypatch.cleanup()
                
                print(f"✓ {test_name} PASSED")
                return True
                
            finally:
                # Restore environment
                if old_db_path is None:
                    os.environ.pop("PRECURSOR_SCRATCHPAD_DB", None)
                else:
                    os.environ["PRECURSOR_SCRATCHPAD_DB"] = old_db_path
                    
    except Exception as e:
        print(f"✗ {test_name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    # Import test functions
    from tests.test_project_transition_integration import (
        test_project_transition_detected,
        test_batch_processing_metrics_and_coalescing,
        test_notification_skipped_when_no_pending_tasks,
        test_notification_sent_when_pending_tasks_exist,
    )
    
    tests = [
        (test_project_transition_detected, "test_project_transition_detected"),
        (test_batch_processing_metrics_and_coalescing, "test_batch_processing_metrics_and_coalescing"),
        (test_notification_skipped_when_no_pending_tasks, "test_notification_skipped_when_no_pending_tasks"),
        (test_notification_sent_when_pending_tasks_exist, "test_notification_sent_when_pending_tasks_exist"),
    ]
    
    print("Running integration tests...")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_func, test_name in tests:
        if run_test(test_func, test_name):
            passed += 1
        else:
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")


if __name__ == "__main__":
    main()