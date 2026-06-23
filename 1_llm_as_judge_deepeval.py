"""
Project 1: LLM-as-Judge (using DeepEval)
------------------------------------------
Tools: deepeval (pytest-native eval framework with custom LLM-as-judge)
"""

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from langchain_ollama import ChatOllama
import json


class OllamaModel(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatOllama(model="llama3.2", temperature=0, format="json")

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "llama3.2"


ollama_llm = OllamaModel()

# Define a custom judge criteria
correctness = GEval(
    name="Correctness",
    criteria="Is the answer factually correct and relevant to the question?",
    evaluation_steps=[
        "Check if the answer directly addresses the question asked.",
        "Check if the answer contains factually accurate information.",
        "Check if the answer is helpful and informative.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
    model=ollama_llm,
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
