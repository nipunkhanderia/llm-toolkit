"""
Project 2: Trajectory Scoring (using ollama library directly)
--------------------------------------------------------------
Tools: ollama (official Python client)
"""

import ollama


def score_trajectory(task, expected_steps, actual_steps):
    """Use Ollama directly to score agent trajectory."""
    response = ollama.chat(model="llama3.2", messages=[{
        "role": "user",
        "content": f"""You are evaluating an AI agent's performance.

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
    }])
    return response["message"]["content"]


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
        print(score_trajectory(task, expected, steps))
        print("-" * 50)
