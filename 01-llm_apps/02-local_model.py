# Local Model Integration Tutorial
# This script demonstrates how to use LOCAL models alongside cloud LLMs
# Learn about "LLM as a Judge" pattern and model comparison techniques

import os
import json
from openai import OpenAI, AzureOpenAI
from IPython.display import Markdown, display
from dotenv import load_dotenv

# =============================================================================
# SETUP: Data Collection for Model Comparison
# =============================================================================

# Lists to store competitors and their responses for comparison
competitors = []  # Model names (e.g., "OpenAI 4o-mini", "Gemini 2.5 Flash")
answers = []      # Their responses to the same question

# This pattern allows us to:
# 1. Collect responses from multiple models
# 2. Compare them systematically
# 3. Use another model to judge the results

# Load environment variables from .env file
load_dotenv()

# This script requires:
# - Cloud LLM APIs (Azure OpenAI, Gemini) for generating responses
# - Local LLM (Ollama) for judging the responses
# - All configured in your .env file
# =============================================================================
# STEP 1: QUESTION GENERATION - Let AI Create the Test Question
# =============================================================================

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Strategy: Let one AI model generate a question, then test other models with it
# This ensures the question isn't biased toward any particular model

# Prompt engineering: Ask for a specific type of question
question_prompt = "Please ask a question to assess someone's IQ. Respond only with the question."
messages = [{"role": "user", "content": question_prompt}]

# Generate the test question
print("🤔 Generating test question with Azure OpenAI...")
response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    messages=messages
)

# Extract the generated question
test_question = response.choices[0].message.content
print(f"📝 Generated Question: {test_question}")
print("\n" + "=" * 60)

# =============================================================================
# STEP 2: COLLECT RESPONSES - Test Multiple Models with Same Question
# =============================================================================

# Now test Azure OpenAI with the generated question
print("🤖 Testing Azure OpenAI with the question...")
messages = [{"role": "user", "content": test_question}]

# Get Azure OpenAI's response to the test question
response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    messages=messages
)

# Store the response for comparison
azureOpenAIResponse = response.choices[0].message.content
competitors.append("OpenAI 4o-mini")      # Model name
answers.append(azureOpenAIResponse)        # Its response

print(f"✅ Azure OpenAI response collected ({len(azureOpenAIResponse)} characters)")
# Test Gemini with the same question
print("🤖 Testing Gemini with the same question...")

# Initialize Gemini client
gemini_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url=os.getenv("GEMINI_API_BASE")
)

# Get Gemini's response to the same test question
response = gemini_client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": test_question}]
)

# Store Gemini's response for comparison
geminiResponse = response.choices[0].message.content
competitors.append("Gemini 2.5 Flash")    # Model name
answers.append(geminiResponse)             # Its response

print(f"✅ Gemini response collected ({len(geminiResponse)} characters)")
print(f"📊 Total models tested: {len(competitors)}")
print("\n" + "=" * 60)

# =============================================================================
# STEP 3: DISPLAY RESPONSES - Review What Each Model Said
# =============================================================================

print("📋 COLLECTED RESPONSES:")
print("=" * 40)

# Display each model's response for human review
# Using zip() to iterate over two lists in parallel - a Pythonic pattern!
# zip() pairs up elements: competitors[0] with answers[0], competitors[1] with answers[1], etc.
for competitor, answer in zip(competitors, answers):
    print(f"🤖 {competitor}:")
    print(f"{answer}")
    print("-" * 40)


# =============================================================================
# STEP 4: PREPARE FOR JUDGMENT - Format Responses for AI Judge
# =============================================================================

print("📄 Preparing responses for AI judge...")

# Combine all responses into a single text for the judge to evaluate
# Using enumerate() to get both index and value - another useful Python pattern!
combined = ""
for index, answer in enumerate(answers):
    combined += f"# Response from competitor {index+1}\n\n"  # Header for each response
    combined += answer + "\n\n"                                  # The actual response

# This creates a formatted document that the judge can easily parse



# =============================================================================
# STEP 5: LLM AS A JUDGE - Use Local Model to Evaluate Cloud Models
# =============================================================================

# Create a comprehensive prompt for the judge
# This demonstrates "prompt engineering" for evaluation tasks
judge_prompt = f"""You are judging a competition between {len(competitors)} competitors.
Each model has been given this question:

{test_question}

Your job is to evaluate each response for clarity and strength of argument, and rank them in order of best to worst.
Respond with JSON, and only JSON, with the following format:
{{"results": [{{"name": "OpenAI 4o-mini"}}, {{"name": "Gemini 2.5 Flash"}}]}}

Here are the responses from each competitor:

{combined}

Now respond with the JSON with the ranked order of the competitors, nothing else. Do not include markdown formatting or code blocks."""

print("🧑‍⚖️ Initializing local AI judge...")

# Initialize Ollama client for LOCAL model
# This demonstrates hybrid approach: cloud models for generation, local for evaluation
ollama = OpenAI(
    base_url=os.getenv("OLLAMA_LOCAL_API_BASE"),  # Local Ollama server
    api_key=os.getenv("OLLAMA_LOCAL_KEY")         # Usually just "ollama"
)

# Use a local model as the judge
# DeepSeek-R1 is good for reasoning tasks like evaluation
judge_model_name = "deepseek-r1:1.5b"
judge_messages = [{"role": "user", "content": judge_prompt}]

print(f"🧑‍⚖️ Asking {judge_model_name} to evaluate responses...")

# Get the judgment from local model
response = ollama.chat.completions.create(
    model=judge_model_name, 
    messages=judge_messages
)

judge_evaluation_result = response.choices[0].message.content

print("\n" + "=" * 60)
print("🏆 FINAL JUDGMENT:")
print("=" * 20)
print(judge_evaluation_result)

# =============================================================================
# LEARNING SUMMARY: LLM AS A JUDGE PATTERN
# =============================================================================
"""
🎆 CONGRATULATIONS! You just implemented the "LLM as a Judge" pattern!

KEY CONCEPTS LEARNED:

1. MODEL COMPARISON METHODOLOGY:
   - Generate test question with one model
   - Test multiple models with the same question
   - Use another model to judge the results objectively

2. HYBRID ARCHITECTURE:
   - Cloud models (Azure OpenAI, Gemini) for generation
   - Local model (Ollama) for evaluation
   - Best of both worlds: powerful cloud + private local

3. DATA COLLECTION PATTERNS:
   - Lists to store competitors and answers
   - zip() for parallel iteration
   - enumerate() for indexed iteration

4. PROMPT ENGINEERING FOR EVALUATION:
   - Structured prompts for consistent evaluation
   - JSON output format for programmatic processing
   - Clear evaluation criteria (clarity, strength of argument)

5. LOCAL MODEL BENEFITS:
   - Privacy: evaluation happens locally
   - Cost: no API charges for judgment
   - Control: you own the evaluation model

6. PYTHON PATTERNS:
   - zip() for pairing lists
   - enumerate() for indexed loops
   - f-strings for dynamic prompt generation
   - List management for data collection

REAL-WORLD APPLICATIONS:
- A/B testing different models
- Automated model evaluation pipelines
- Quality assurance for AI systems
- Research and benchmarking

TRY THIS:
- Add more models to the competition
- Try different judge models
- Experiment with different evaluation criteria
- Save results to files for analysis

NEXT STEPS:
- Set up Ollama locally to run this script
- Explore other local models for different tasks
- Build evaluation pipelines for your use cases
"""
