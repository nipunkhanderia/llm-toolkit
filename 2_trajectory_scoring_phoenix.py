"""
Project 2: Trajectory Scoring (using Arize Phoenix)
-----------------------------------------------------
Tools: openinference (OpenTelemetry-based LLM tracing) + Phoenix UI

HOW TO RUN:
1. Start Phoenix server first (pick one):
   Option A: pip install arize-phoenix && python -m phoenix.server.main serve
   Option B: docker run -p 6006:6006 arizephoenix/phoenix:latest

2. Then run this script:
   python 2_trajectory_scoring_phoenix.py

3. Open http://localhost:6006 to see traced trajectory
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Send traces to Phoenix running at localhost:6006
endpoint = "http://localhost:6006/v1/traces"
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
trace.set_tracer_provider(provider)

# Instrument LangChain — auto-captures every LLM call as a span
LangChainInstrumentor().instrument(tracer_provider=provider)

llm = ChatOllama(model="llama3.2", temperature=0)


def agent_step(step_name, prompt_text):
    """Simulate an agent step — each call is traced as a span in Phoenix."""
    prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
    chain = prompt | llm
    result = chain.invoke({"input": prompt_text})
    print(f"  ✓ {step_name}: {result.content[:80]}...")
    return result.content


if __name__ == "__main__":
    task = "Book a flight from NYC to London"
    print(f"Task: {task}")
    print("=" * 60)
    print("Running agent steps (traced to Phoenix at localhost:6006)...\n")

    agent_step("search_flights", f"List 3 flights from NYC to London with prices. Task: {task}")
    agent_step("select_flight", "Select the cheapest flight from: Flight A $400, Flight B $350, Flight C $500")
    agent_step("enter_details", "Fill passenger details: name John Doe, passport XX123456")
    agent_step("payment", "Process payment of $350 for Flight B using credit card ending 4242")
    agent_step("confirm", "Confirm booking for Flight B, NYC to London, passenger John Doe")

    print("\n" + "=" * 60)
    print("✅ All steps traced!")
    print("📊 Open http://localhost:6006 to view the full trajectory tree")
