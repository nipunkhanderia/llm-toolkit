"""
Project 1: LLM-as-Judge (using DeepEval)
------------------------------------------
Tools: deepeval (pytest-native eval framework with GEval LLM-as-judge)
"""

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.metrics.indicator import ScoreType

# Define a custom judge criteria
correctness = GEval(
    name="Correctness",
    criteria="Is the answer factually correct and relevant to the question?",
    evaluation_params=["input", "actual_output"],
    threshold=0.5,
    model="ollama/llama3.2",
)

# Create test cases (question + answer pairs to judge)
test_cases = [
    LLMTestCase(
        input="What is Python?",
        actual_output="Python is a high-level programming language known for readability and versatility.",
    ),
    LLMTestCase(
        input="What is Python?",
        actual_output="idk",
    ),
    LLMTestCase(
        input="What is Python?",
        actual_output="A snake that eats mice.",
    ),
]

if __name__ == "__main__":
    print("🧑‍⚖️ LLM-as-Judge with DeepEval GEval\n")
    evaluate(test_cases, [correctness])
