"""
Project 2: Trajectory Scoring (using Arize Phoenix)
-----------------------------------------------------
Tools: arize-phoenix (OpenTelemetry-based LLM tracing)
Records every agent step as a span and visualises the trajectory.
"""

import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Launch Phoenix tracing (opens UI at http://localhost:6006)
px.launch_app()
tracer_provider = register()
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

llm = ChatOllama(model="llama3.2", temperature=0)


def agent_step(step_name, prompt_text):
    """Simulate an agent step — each call is traced as a span in Phoenix."""
    prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
    chain = prompt | llm
    result = chain.invoke({"input": prompt_text})
    print(f"  Step: {step_name} → {result.content[:80]}...")
    return result.content


if __name__ == "__main__":
    task = "Book a flight from NYC to London"
    print(f"Task: {task}\n")
    print("Running agent steps (each traced in Phoenix)...\n")

    # Simulate agent trajectory — each step becomes a traced span
    agent_step("search_flights", f"List 3 flights from NYC to London with prices. Task: {task}")
    agent_step("select_flight", "Select the cheapest flight from the options: Flight A $400, Flight B $350, Flight C $500")
    agent_step("enter_details", "Fill passenger details: name John Doe, passport XX123456")
    agent_step("payment", "Process payment of $350 for Flight B using credit card ending 4242")
    agent_step("confirm", "Confirm booking for Flight B, NYC to London, passenger John Doe")

    print("\n✅ All steps traced!")
    print("📊 Open Phoenix UI at http://localhost:6006 to view the full trajectory")
    print("   You'll see each step as a span with latency, input/output, and token counts")
    input("\nPress Enter to exit...")
