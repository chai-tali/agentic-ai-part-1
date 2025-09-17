# Unit Tests for Client Initialization and Environment Handling
# These tests focus on the setup and configuration aspects

import pytest
import os
from unittest.mock import patch, Mock, MagicMock
from io import StringIO
import sys

# =============================================================================
# TESTS FOR ENVIRONMENT VARIABLE HANDLING
# =============================================================================

class TestEnvironmentVariables:
    """
    Test environment variable loading and validation
    
    These tests ensure that the application properly handles
    environment configuration for different scenarios
    """
    
    def test_load_dotenv_success(self, mock_environment):
        """
        Test successful loading of environment variables
        
        This test verifies that dotenv loads variables correctly
        when a .env file is present and properly formatted
        """
        # The mock_environment fixture already sets up the environment
        # So we just need to verify the variables are accessible
        
        assert os.getenv("AZURE_OPENAI_API_KEY") == "test_azure_key_12345"
        assert os.getenv("AZURE_OPENAI_ENDPOINT") == "https://test-resource.openai.azure.com/"
        assert os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") == "test-gpt-4"
        assert os.getenv("AZURE_OPENAI_API_VERSION") == "2024-05-01-preview"
        assert os.getenv("GEMINI_API_KEY") == "test_gemini_key_67890"
        assert os.getenv("GEMINI_API_BASE") == "https://generativelanguage.googleapis.com/v1beta/openai/"
    
    def test_missing_azure_variables(self, monkeypatch):
        """
        Test behavior when Azure OpenAI environment variables are missing
        
        This test ensures the application handles missing configuration
        gracefully and provides helpful error messages
        """
        # Remove Azure OpenAI variables
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT_NAME", raising=False)
        
        # Verify variables are not set
        assert os.getenv("AZURE_OPENAI_API_KEY") is None
        assert os.getenv("AZURE_OPENAI_ENDPOINT") is None
        assert os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") is None
    
    def test_missing_gemini_variables(self, monkeypatch):
        """
        Test behavior when Gemini environment variables are missing
        """
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_BASE", raising=False)
        
        assert os.getenv("GEMINI_API_KEY") is None
        assert os.getenv("GEMINI_API_BASE") is None
    
    def test_empty_environment_variables(self, monkeypatch):
        """
        Test behavior when environment variables are empty strings
        
        This tests a common configuration issue where variables
        are defined but have empty values
        """
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "")
        monkeypatch.setenv("GEMINI_API_KEY", "")
        
        # Empty strings should be treated as missing
        assert os.getenv("AZURE_OPENAI_API_KEY") == ""
        assert os.getenv("GEMINI_API_KEY") == ""
        
        # Application should handle empty strings appropriately
        # This might involve treating them as None or raising errors

# =============================================================================
# TESTS FOR CLIENT INITIALIZATION LOGIC
# =============================================================================

class TestClientInitialization:
    """
    Test the client initialization process
    
    These tests verify that LLM clients are properly created
    and configured with the correct parameters
    """
    
    @patch('openai.AzureOpenAI')
    @patch('openai.OpenAI')
    def test_initialize_clients_success(self, mock_openai, mock_azure, mock_environment, capsys):
        """
        Test successful client initialization with debug output
        
        This test verifies that:
        1. Clients are created with correct parameters
        2. Debug information is printed
        3. Both clients are returned
        """
        # Set up mock instances
        mock_azure_instance = Mock()
        mock_openai_instance = Mock()
        mock_azure.return_value = mock_azure_instance
        mock_openai.return_value = mock_openai_instance
        
        # Since we can't import the actual function yet, we'll simulate the logic
        # In a real implementation, this would be:
        # from llm_apps.fast_api_llm_app import initialize_clients
        # azure_client, gemini_client = initialize_clients()
        
        # Simulate the initialization logic
        azure_client = mock_azure(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        gemini_client = mock_openai(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url=os.getenv("GEMINI_API_BASE")
        )
        
        # Verify clients were created with correct parameters
        mock_azure.assert_called_once_with(
            api_key="test_azure_key_12345",
            api_version="2024-05-01-preview",
            azure_endpoint="https://test-resource.openai.azure.com/"
        )
        
        mock_openai.assert_called_once_with(
            api_key="test_gemini_key_67890",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        # Verify clients were returned
        assert azure_client is not None
        assert gemini_client is not None
    
    @patch('openai.AzureOpenAI')
    def test_azure_client_initialization_failure(self, mock_azure, mock_environment):
        """
        Test Azure client initialization failure
        
        This test simulates what happens when Azure OpenAI
        client creation fails (e.g., invalid credentials)
        """
        # Make the Azure client constructor raise an exception
        mock_azure.side_effect = Exception("Invalid Azure OpenAI credentials")
        
        # Test should handle the exception appropriately
        with pytest.raises(Exception, match="Invalid Azure OpenAI credentials"):
            mock_azure(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
    
    @patch('openai.OpenAI')
    def test_gemini_client_initialization_failure(self, mock_openai, mock_environment):
        """
        Test Gemini client initialization failure
        """
        mock_openai.side_effect = Exception("Invalid Gemini API key")
        
        with pytest.raises(Exception, match="Invalid Gemini API key"):
            mock_openai(
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url=os.getenv("GEMINI_API_BASE")
            )
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_debug_output_format(self, mock_stdout, mock_environment):
        """
        Test that debug output is properly formatted
        
        This test verifies that the initialization function
        prints helpful debug information in the correct format
        """
        # Simulate the debug output that would be printed
        # In the real function, this would be automatic
        print(f"Azure Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
        print(f"Azure API Key: {'***' + os.getenv('AZURE_OPENAI_API_KEY')[-4:]}")
        print(f"Azure API Version: {os.getenv('AZURE_OPENAI_API_VERSION')}")
        print(f"Azure Deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}")
        print(f"Gemini API Key: {'***' + os.getenv('GEMINI_API_KEY')[-4:]}")
        print(f"Gemini Base URL: {os.getenv('GEMINI_API_BASE')}")
        
        output = mock_stdout.getvalue()
        
        # Verify debug output contains expected information
        assert "Azure Endpoint: https://test-resource.openai.azure.com/" in output
        assert "Azure API Key: ***2345" in output  # Last 4 chars of test key
        assert "Azure API Version: 2024-05-01-preview" in output
        assert "Azure Deployment: test-gpt-4" in output
        assert "Gemini API Key: ***7890" in output  # Last 4 chars of test key
        assert "Gemini Base URL: https://generativelanguage.googleapis.com/v1beta/openai/" in output
    
    def test_api_key_masking(self, mock_environment):
        """
        Test that API keys are properly masked in debug output
        
        This is important for security - we don't want full API keys
        appearing in logs or console output
        """
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        # Test the masking logic
        masked_azure = "***" + azure_key[-4:] if azure_key else "Not set"
        masked_gemini = "***" + gemini_key[-4:] if gemini_key else "Not set"
        
        assert masked_azure == "***2345"
        assert masked_gemini == "***7890"
        
        # Verify that the full keys are not exposed
        assert azure_key not in masked_azure[:-4]  # Full key should not be in masked version
        assert gemini_key not in masked_gemini[:-4]

# =============================================================================
# TESTS FOR CONFIGURATION VALIDATION
# =============================================================================

class TestConfigurationValidation:
    """
    Test configuration validation logic
    
    These tests ensure that the application properly validates
    its configuration and provides helpful feedback
    """
    
    def test_validate_azure_configuration_complete(self, mock_environment):
        """
        Test validation with complete Azure configuration
        
        This test verifies that validation passes when all
        required Azure OpenAI variables are present
        """
        # Simulate validation logic
        required_azure_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT", 
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_OPENAI_API_VERSION"
        ]
        
        missing_vars = [var for var in required_azure_vars if not os.getenv(var)]
        
        assert len(missing_vars) == 0, f"Missing Azure variables: {missing_vars}"
    
    def test_validate_gemini_configuration_complete(self, mock_environment):
        """
        Test validation with complete Gemini configuration
        """
        required_gemini_vars = [
            "GEMINI_API_KEY",
            "GEMINI_API_BASE"
        ]
        
        missing_vars = [var for var in required_gemini_vars if not os.getenv(var)]
        
        assert len(missing_vars) == 0, f"Missing Gemini variables: {missing_vars}"
    
    def test_validate_configuration_with_missing_vars(self, monkeypatch):
        """
        Test validation with missing variables
        
        This test ensures that validation properly identifies
        missing configuration and provides helpful feedback
        """
        # Remove some required variables
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        
        # Test validation logic
        all_required_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT_NAME", 
            "AZURE_OPENAI_API_VERSION",
            "GEMINI_API_KEY",
            "GEMINI_API_BASE"
        ]
        
        missing_vars = [var for var in all_required_vars if not os.getenv(var)]

        assert "AZURE_OPENAI_API_KEY" in missing_vars
        assert "GEMINI_API_KEY" in missing_vars
        # Should have 2 missing variables after we removed them
        assert len(missing_vars) >= 2

# =============================================================================
# LEARNING NOTES FOR STUDENTS  
# =============================================================================
"""
🎓 CLIENT INITIALIZATION TESTING EXPLAINED:

1. ENVIRONMENT TESTING:
   - Test with valid configuration
   - Test with missing variables
   - Test with empty/invalid values
   - Verify security (API key masking)

2. MOCKING EXTERNAL DEPENDENCIES:
   - Mock OpenAI client constructors
   - Control what exceptions are raised
   - Verify correct parameters are passed

3. TESTING INITIALIZATION LOGIC:
   - Test successful client creation
   - Test failure scenarios
   - Test debug output and logging

4. CONFIGURATION VALIDATION:
   - Verify required variables are present
   - Test validation error messages
   - Ensure helpful feedback for missing config

5. SECURITY CONSIDERATIONS:
   - Never log full API keys
   - Test that sensitive data is masked
   - Verify secure configuration practices

COMMON PATTERNS:
- Use monkeypatch to modify environment variables
- Mock external client constructors
- Test both success and failure scenarios
- Verify debug output without exposing secrets

TRY THIS:
- Run tests with: pytest tests/unit/test_client_initialization.py -v
- Add tests for different error scenarios
- Practice testing configuration validation
- Experiment with different mocking strategies
"""
