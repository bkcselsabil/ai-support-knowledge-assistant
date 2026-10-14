
from langgraph_test import app

test_cases = [
    {
        "question": "What is 50 minus 18?",
        "expected_route": "calculator",
        "expected_fact": "32",
    },
    {
        "question": "How many vacation days do employees get?",
        "expected_route": "rag",
        "expected_fact": "30",
    },
    {
        "question": "What is the company's policy on pets in the office?",
        "expected_route": "rag",
        "expected_fact": "couldn't find relevant information",
    },
    {
        "question": "Can employees work remotely?",
        "expected_route": "rag",
        "expected_fact": "three days per week",
    },
    {
    "question": "What is 24 multiplied by 3?",
    "expected_route": "calculator",
    "expected_fact": "72",
},
    {
        "question": "An employee has taken 12 vacation days. How many vacation days do they have left?",
        "expected_route": "rag",
        "expected_fact": "18",
    },

]

passed_tests = 0

for test in test_cases:
    result = app.invoke({
        "message": test["question"],
        "result": "",
        "route": "",
    })

    route_passed = result["route"] == test["expected_route"]
    answer_passed = test["expected_fact"] in result["result"]

    passed = route_passed and answer_passed

    if passed:
        passed_tests += 1

    print("\nQuestion:", test["question"])
    print("Expected route:", test["expected_route"])
    print("Actual route:", result["route"])
    print("Answer:", result["result"])
    print("Route:", "PASS" if route_passed else "FAIL")
    print("Answer:", "PASS" if answer_passed else "FAIL")
    print("Result:", "PASS" if passed else "FAIL")

print(f"\nPassed: {passed_tests}/{len(test_cases)}")
