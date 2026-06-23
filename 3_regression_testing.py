"""
Project 3: Automated Regression Testing
-----------------------------------------
Uses pytest for testing + Ollama (llama3.2) to auto-generate test cases.
Tools: pytest, langchain, ollama
"""

import json
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0)


# --- Functions to test ---
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return "error"
    return a / b


def generate_test_cases(func_name, func_description):
    """Use LLM to generate test cases for a function."""
    prompt = f"""Generate 3 test cases for this function as JSON array.
Function: {func_name}
Description: {func_description}

Respond ONLY with a JSON array like:
[{{"inputs": [1, 2], "expected": 3}}, {{"inputs": [0, 0], "expected": 0}}]"""

    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        text = response.content
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
        return []


def run_tests(func, test_cases):
    """Run test cases against a function."""
    results = []
    for test in test_cases:
        inputs = test["inputs"]
        expected = test["expected"]
        actual = func(*inputs)
        passed = actual == expected
        results.append({
            "inputs": inputs,
            "expected": expected,
            "actual": actual,
            "passed": passed,
        })
    return results


if __name__ == "__main__":
    functions_to_test = [
        {"name": "add", "func": add, "desc": "adds two numbers: add(a, b) -> a + b"},
        {"name": "multiply", "func": multiply, "desc": "multiplies two numbers: multiply(a, b) -> a * b"},
        {"name": "divide", "func": divide, "desc": "divides two numbers: divide(a, b) -> a/b, returns 'error' if b is 0"},
    ]

    print("🧪 AI-Powered Regression Testing\n")

    for item in functions_to_test:
        print(f"Function: {item['name']}()")
        print(f"  Generating test cases with LLM...")
        test_cases = generate_test_cases(item["name"], item["desc"])
        print(f"  Generated {len(test_cases)} tests")

        results = run_tests(item["func"], test_cases)
        for r in results:
            status = "✓ PASS" if r["passed"] else "✗ FAIL"
            print(f"    {status}: {item['name']}{tuple(r['inputs'])} = {r['actual']} (expected {r['expected']})")

        passed = sum(1 for r in results if r["passed"])
        print(f"  Result: {passed}/{len(results)} passed\n")
