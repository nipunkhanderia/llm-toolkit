"""
Project 3: Automated Regression Testing (using ollama library directly)
------------------------------------------------------------------------
Tools: ollama (official Python client), pytest
"""

import json
import ollama


def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return "error"
    return a / b


def generate_test_cases(func_name, func_description):
    """Use Ollama directly to generate test cases."""
    response = ollama.chat(model="llama3.2", messages=[{
        "role": "user",
        "content": f"""Generate 3 test cases for this function as JSON array.
Function: {func_name}
Description: {func_description}

Respond ONLY with a JSON array like:
[{{"inputs": [1, 2], "expected": 3}}, {{"inputs": [0, 0], "expected": 0}}]"""
    }])
    text = response["message"]["content"]
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end != 0:
        return json.loads(text[start:end])
    return []


if __name__ == "__main__":
    functions_to_test = [
        {"name": "add", "func": add, "desc": "adds two numbers: add(a, b) -> a + b"},
        {"name": "multiply", "func": multiply, "desc": "multiplies two numbers: multiply(a, b) -> a * b"},
        {"name": "divide", "func": divide, "desc": "divides two numbers: divide(a, b) -> a/b, returns 'error' if b is 0"},
    ]

    print("🧪 AI-Powered Regression Testing (Ollama)\n")

    for item in functions_to_test:
        print(f"Function: {item['name']}()")
        print(f"  Generating test cases with LLM...")
        test_cases = generate_test_cases(item["name"], item["desc"])
        print(f"  Generated {len(test_cases)} tests")

        for test in test_cases:
            actual = item["func"](*test["inputs"])
            passed = actual == test["expected"]
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"    {status}: {item['name']}{tuple(test['inputs'])} = {actual} (expected {test['expected']})")
        print()
