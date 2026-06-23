"""
Project 1: LLM-as-Judge
-------------------------
Uses Ollama (llama3.2) as a judge to score AI-generated answers.
Tools: langchain, ollama
"""

from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0)

def judge_answer(question, answer):
    """Use LLM to judge the quality of an answer."""
    prompt = f"""You are a strict judge. Score the following answer on a scale of 1-10.
Provide: score, reasoning (1 sentence).

Question: {question}
Answer: {answer}

Respond in this exact format:
Score: <number>
Reasoning: <one sentence>"""

    response = llm.invoke(prompt)
    return response.content


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
        result = judge_answer(question, ans)
        print(f"  Judge says: {result}\n")
