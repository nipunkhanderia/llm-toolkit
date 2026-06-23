"""
Project 1: LLM-as-Judge (using ollama library directly)
--------------------------------------------------------
Tools: ollama (official Python client)
"""

import ollama


def judge_answer(question, answer):
    """Use Ollama directly to judge answer quality."""
    response = ollama.chat(model="llama3.2", messages=[{
        "role": "user",
        "content": f"""You are a strict judge. Score the following answer on a scale of 1-10.

Question: {question}
Answer: {answer}

Respond in this exact format:
Score: <number>
Reasoning: <one sentence>"""
    }])
    return response["message"]["content"]


if __name__ == "__main__":
    question = "What is Python?"
    answers = [
        "Python is a high-level programming language known for its readability and versatility.",
        "idk",
        "A snake that eats mice.",
    ]

    print(f"Question: {question}\n")
    for i, ans in enumerate(answers, 1):
        print(f"Answer {i}: \"{ans}\"")
        print(f"  Judge says: {judge_answer(question, ans)}\n")
