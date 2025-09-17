# 🧪 Testing Documentation

This directory contains the comprehensive test suite for the LLM Apps project.

## 🏃‍♂️ Quick Start

For complete setup and testing instructions, see the **[main project README](../README.md#-testing-guide)**.

### Run All Tests
```bash
# From project root
uv run pytest tests
```

### Start FastAPI Server (for manual testing)
```bash
# From project root
uv run uvicorn 01-llm_apps.04-fast_api_llm_app:app --reload
```

## 📁 Test Structure

```
tests/
├── README.md                     # This file - testing overview
├── conftest.py                   # Pytest fixtures and configuration
├── unit/                         # Unit tests (fast, isolated)
│   ├── test_fastapi_functions.py # Test individual functions
│   └── test_client_initialization.py # Test setup and config
├── integration/                  # Integration tests (end-to-end)
│   └── test_api_endpoints.py     # Test FastAPI endpoints
└── manual/                       # Manual/interactive tests
    └── test_api_client.py         # Manual API testing script
```

## 🎯 Test Categories

| Type | Location | Purpose | Speed |
|------|----------|---------|-------|
| **Unit** | `tests/unit/` | Individual functions | Fast (< 1s) |
| **Integration** | `tests/integration/` | End-to-end API | Medium (1-5s) |
| **Manual** | `tests/manual/` | Interactive learning | Variable |

## 🧪 Running Specific Tests

```bash
# Unit tests only (fast)
uv run pytest tests/unit/ -v

# Integration tests only
uv run pytest tests/integration/ -v

# Specific test file
uv run pytest tests/unit/test_fastapi_functions.py -v

# With coverage report
uv run pytest tests --cov=01-llm_apps --cov-report=html
```

## 📚 Learning Resources

### For Complete Documentation:
- **[Main README](../README.md)** - Full project setup and testing guide
- **Code Comments** - Every test function is documented
- **Learning Notes** - Educational comments in each test file

### Test-Specific Learning:
- **Unit Tests**: Learn function isolation and mocking
- **Integration Tests**: Understand API testing patterns
- **Manual Tests**: Interactive testing and debugging

## 🔍 What Each Test File Covers

### `unit/test_fastapi_functions.py`
- Individual function testing
- Mocking external API calls
- Error handling verification
- Input/output validation

### `unit/test_client_initialization.py`
- Environment setup testing
- Client configuration
- Dependency validation

### `integration/test_api_endpoints.py`
- Full FastAPI endpoint testing
- Request/response validation
- Error handling end-to-end
- Real HTTP request simulation

### `manual/test_api_client.py`
- Interactive API testing
- Real-time streaming demos
- Troubleshooting utilities
- Learning examples

## 🎓 Educational Notes

Each test file contains extensive learning notes and comments explaining:
- **Why** each test exists
- **How** the testing patterns work
- **When** to use different testing approaches
- **Best practices** for LLM app testing

Happy Testing! 🚀