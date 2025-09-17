# Basic Test to Verify Testing Framework
# Simple tests to ensure pytest is working correctly

import pytest
import os
import sys
from pathlib import Path

def test_pytest_working():
    """
    Basic test to verify pytest is functioning
    
    This test should always pass and confirms that:
    1. Pytest can discover and run tests
    2. Basic assertions work
    3. The testing framework is properly configured
    """
    assert True
    assert 1 + 1 == 2
    assert "hello" == "hello"

def test_environment_setup():
    """
    Test that we're running in the correct environment
    
    This test verifies:
    1. We're in the right directory
    2. Required files exist
    3. Python version is compatible
    """
    # Check we're in the project directory
    current_dir = Path.cwd()
    assert current_dir.name == "agentic-ai-part-1"
    
    # Check key files exist
    assert (current_dir / "pyproject.toml").exists()
    assert (current_dir / "01-llm_apps").exists()
    assert (current_dir / "tests").exists()
    
    # Check Python version
    assert sys.version_info >= (3, 10)

def test_project_structure():
    """
    Test that the project structure is correct
    
    This verifies that all expected files and directories exist
    """
    project_root = Path.cwd()
    
    # Check main directories
    assert (project_root / "01-llm_apps").is_dir()
    assert (project_root / "tests").is_dir()
    
    # Check test directories
    tests_dir = project_root / "tests"
    assert (tests_dir / "unit").is_dir()
    assert (tests_dir / "integration").is_dir()
    assert (tests_dir / "manual").is_dir()
    
    # Check key files
    assert (project_root / "pyproject.toml").is_file()
    assert (project_root / "pytest.ini").is_file()
    assert (tests_dir / "conftest.py").is_file()

def test_llm_apps_directory_contents():
    """
    Test that the 01-llm_apps directory has the expected files
    """
    llm_apps_dir = Path.cwd() / "01-llm_apps"
    
    # Check for main Python files (some might be missing due to renames)
    expected_files = [
        "04-fast_api_llm_app.py",
        "01-openai_sdk_foundation.py",
    ]
    
    for file_name in expected_files:
        file_path = llm_apps_dir / file_name
        if file_path.exists():
            assert file_path.is_file()
            # Check file is not empty
            assert file_path.stat().st_size > 0

@pytest.mark.parametrize("test_value,expected", [
    (1, 1),
    ("hello", "hello"),
    ([1, 2, 3], [1, 2, 3]),
    ({"key": "value"}, {"key": "value"}),
])
def test_parametrized_example(test_value, expected):
    """
    Example of parametrized testing
    
    This demonstrates how pytest can run the same test
    with different inputs and expected outputs
    """
    assert test_value == expected

def test_fixtures_basic(mock_env_vars):
    """
    Test that basic fixtures work
    
    This test uses a fixture from conftest.py to verify
    that the fixture system is working correctly
    """
    # The mock_env_vars fixture should provide test environment variables
    assert isinstance(mock_env_vars, dict)
    assert "AZURE_OPENAI_API_KEY" in mock_env_vars
    assert "GEMINI_API_KEY" in mock_env_vars

class TestClassExample:
    """
    Example test class to demonstrate test organization
    
    Test classes help organize related tests together
    and can share common setup/teardown logic
    """
    
    def test_class_method_example(self):
        """Test method within a class"""
        assert "test" in "testing"
    
    def test_another_class_method(self):
        """Another test method in the same class"""
        result = self.helper_method()
        assert result == "helper_result"
    
    def helper_method(self):
        """Helper method for tests (not a test itself)"""
        return "helper_result"

# =============================================================================
# LEARNING NOTES FOR STUDENTS
# =============================================================================
"""
🎓 BASIC TESTING CONCEPTS DEMONSTRATED:

1. TEST DISCOVERY:
   - Pytest finds files matching test_*.py pattern
   - Functions starting with test_ are automatically run
   - Classes starting with Test are test classes

2. ASSERTIONS:
   - Use assert statements to verify expected behavior
   - assert condition, "optional error message"
   - Many assertion helpers available

3. TEST ORGANIZATION:
   - Group related tests in classes
   - Use descriptive names for tests and classes
   - One concept per test function

4. PARAMETRIZED TESTS:
   - @pytest.mark.parametrize runs same test with different data
   - Reduces code duplication
   - Tests multiple scenarios efficiently

5. FIXTURES:
   - Reusable test setup and data
   - Defined in conftest.py or test files
   - Automatically injected into test functions

6. TEST STRUCTURE:
   - Arrange: Set up test data and conditions
   - Act: Execute the code being tested
   - Assert: Verify the results

TRY THIS:
- Run: pytest tests/test_basic.py -v
- Add your own test functions
- Experiment with different assertions
- Create your own fixtures
"""
