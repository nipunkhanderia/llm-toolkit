"""
Project 1: LLM-as-Judge (using LiteLLM)
-----------------------------------------
Tools: litellm (unified API for 100+ LLM providers)
"""

from litellm import completion


def judge_answer(question, answer):
    """Use LiteLLM to judge answer quality via Ollama."""
    response = completion(model="ollama/llama3.2", messages=[{
        "role": "user",
        "content": f"""You are a strict judge. Score the following answer on a scale of 1-10.

Question: {question}
Answer: {answer}

Respond in this exact format:
Score: <number>
Reasoning: <one sentence>"""
    }], api_base="http://localhost:11434")
    return response.choices[0].message.content


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
