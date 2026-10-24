from examples.langgraph_test import calculate


def test_subtraction():
    assert calculate(50, "subtract", 18) == 32


def test_addition():
    assert calculate(25, "add", 17) == 42


def test_multiplication():
    assert calculate(12, "multiply", 9) == 108


def test_division():
    assert calculate(144, "divide", 12) == 12


def test_division_by_zero():
    assert calculate(10, "divide", 0) == "Error: Cannot divide by zero"


def test_unsupported_operation():
    result = calculate(5, "power", 2)

    assert result == "Error: Unsupported operation."
