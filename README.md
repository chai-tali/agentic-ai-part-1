# Agentic AI Project Setup

This repository contains examples and exercises for working with Agentic AI capabilities using Python.

## Prerequisites
- Python 3.10+ recommended
- UV package manager (recommended) or pip

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/ximanta/agentic-ai-part-1.git
cd agentic-ai-part-1
```

2. Initialize the project 
**Option A: Using uv (recommended)**
```bash
uv venv
uv sync
```
**Option B: Using pip**
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install .
```

3. Configure environment variables:
    - Copy the `.envbackup` file in the root directory and rename it to `.env`
   - Fill in your API keys and endpoints

## Running the Examples

1. Basic OpenAI SDK Example:
```bash
uv run python 01-llm_apps/01-openai_sdk_foundation.py
```



## Troubleshooting

1. **Environment Variables**: 
   - Ensure your `.env` file exists in the root directory
   - Verify your LLM API keys are correct
   - Make sure python-dotenv is installed and working

2. **Dependencies**: 
   If you encounter package-related issues, try:
   ```bash
     rm -rf .venv
     uv venv && uv sync
   ```
   


3. **API Access**: 
   - Verify your LLM  API key is valid
   - Check if you have sufficient API credits
   - Ensure you're not hitting rate limits
