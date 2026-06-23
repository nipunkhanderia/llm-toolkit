"""
Project 3: Regression Testing (using RAGAS)
---------------------------------------------
Tools: ragas (RAG evaluation metrics: faithfulness, answer correctness, context precision)
"""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from datasets import Dataset

# Setup Ollama as the evaluator LLM
eval_llm = LangchainLLMWrapper(ChatOllama(model="llama3.2", temperature=0))
eval_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="llama3.2"))

# Sample RAG pipeline outputs to evaluate
data = {
    "question": [
        "What is Python?",
        "What is Python?",
        "Who created Python?",
    ],
    "answer": [
        "Python is a high-level programming language created by Guido van Rossum.",
        "Python is a type of snake found in Asia.",
        "Python was created by Guido van Rossum in 1991.",
    ],
    "contexts": [
        ["Python is a high-level, general-purpose programming language created by Guido van Rossum in 1991."],
        ["Python is a high-level, general-purpose programming language created by Guido van Rossum in 1991."],
        ["Guido van Rossum began working on Python in 1989 and released version 0.9.0 in 1991."],
    ],
    "ground_truth": [
        "Python is a high-level programming language created by Guido van Rossum.",
        "Python is a high-level programming language created by Guido van Rossum.",
        "Guido van Rossum created Python in 1991.",
    ],
}

dataset = Dataset.from_dict(data)

if __name__ == "__main__":
    print("🧪 RAG Regression Testing with RAGAS\n")
    print("Evaluating RAG outputs on: faithfulness, answer_correctness, context_precision\n")

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_correctness, context_precision],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    print(result)
    print("\n📊 Results DataFrame:")
    df = result.to_pandas()
    print(df.to_string(index=False))
