"""
Project 2: Trajectory Scoring
-------------------------------
Scores a sequence of agent steps using Ollama (llama3.2).
Tools: langchain, ollama
"""

from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0)

def score_trajectory(task, expected_steps, actual_steps):
    """Use LLM to evaluate an agent's trajectory."""
    prompt = f"""You are evaluating an AI agent's performance.

Task: {task}
Expected steps: {expected_steps}
Actual steps taken: {actual_steps}

Score the trajectory from 1-10 based on:
- Correctness: Did it follow the right steps?
- Efficiency: Were there unnecessary steps?
- Completeness: Were all required steps done?

Respond in this exact format:
Score: <number>/10
Correctness: <brief note>
Efficiency: <brief note>
Completeness: <brief note>"""

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    task = "Book a flight from NYC to London"
    expected = ["search flights", "select flight", "enter passenger details", "pay", "confirm booking"]

    trajectories = {
        "Good Agent": ["search flights", "select flight", "enter passenger details", "pay", "confirm booking"],
        "Confused Agent": ["search hotels", "search flights", "select flight", "enter details", "pay", "confirm"],
        "Lazy Agent": ["search flights", "select flight"],
    }

    print(f"Task: {task}")
    print(f"Expected: {expected}\n")
    print("=" * 50)

    for agent_name, steps in trajectories.items():
        print(f"\n🤖 {agent_name}: {steps}")
        result = score_trajectory(task, expected, steps)
        print(result)
        print("-" * 50)
