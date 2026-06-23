"""
Pytest regression tests — run with: pytest test_functions.py -v
"""
import pytest


# --- Functions under test (same as in project 3) ---
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return "error"
    return a / b


# --- Tests ---
class TestAdd:
    def test_positive(self):
        assert add(2, 3) == 5

    def test_negative(self):
        assert add(-1, -1) == -2

    def test_zero(self):
        assert add(0, 0) == 0


class TestMultiply:
    def test_basic(self):
        assert multiply(3, 4) == 12

    def test_zero(self):
        assert multiply(0, 100) == 0

    def test_negative(self):
        assert multiply(-2, 3) == -6


class TestDivide:
    def test_basic(self):
        assert divide(10, 2) == 5.0

    def test_division_by_zero(self):
        assert divide(7, 0) == "error"

    def test_float_result(self):
        assert divide(7, 2) == 3.5
