# 3 Python Projects: LLM Evaluation & Testing

## Setup

### 1. Install Ollama
Download from https://ollama.com and install it.

Then pull the model:
```bash
ollama pull llama3.2
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

## Run the Projects

```bash
# Project 1: LLM-as-Judge
python 1_llm_as_judge.py              # LangChain
python 1_llm_as_judge_ollama.py       # Ollama (direct)
python 1_llm_as_judge_litellm.py      # LiteLLM
python 1_llm_as_judge_deepeval.py     # DeepEval GEval

# Project 2: Trajectory Scoring
python 2_trajectory_scoring.py             # LangChain
python 2_trajectory_scoring_ollama.py      # Ollama (direct)
python 2_trajectory_scoring_litellm.py     # LiteLLM
python 2_trajectory_scoring_phoenix.py     # Arize Phoenix

# Project 3: Regression Testing
python 3_regression_testing.py             # LangChain
python 3_regression_testing_ollama.py      # Ollama (direct)
python 3_regression_testing_litellm.py     # LiteLLM
python 3_regression_testing_ragas.py       # RAGAS

# Pytest tests
pytest test_functions.py -v
```

## Tools Used

| Project | Tool | What it does |
|---------|------|-------------|
| LLM-as-Judge | LangChain | Simple LLM call to judge |
| LLM-as-Judge | Ollama (direct) | Official Python client, minimal |
| LLM-as-Judge | LiteLLM | Unified API for 100+ providers |
| LLM-as-Judge | **DeepEval** ⭐ | pytest-native GEval judge with custom criteria |
| Trajectory | LangChain | LLM scores trajectory |
| Trajectory | Ollama (direct) | Direct API call |
| Trajectory | LiteLLM | Provider-agnostic |
| Trajectory | **Arize Phoenix** ⭐ | OpenTelemetry tracing, visual trajectory UI |
| Regression | LangChain | LLM generates test cases |
| Regression | Ollama (direct) | Direct API call |
| Regression | LiteLLM | Provider-agnostic |
| Regression | **RAGAS** ⭐ | RAG metrics: faithfulness, correctness, precision |
