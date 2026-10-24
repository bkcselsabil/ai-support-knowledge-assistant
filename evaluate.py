from agent import run_agent
from rag import retrieve_documents, generate_answer, collection

test_cases = [
    {
        "question": "How many vacation days do employees get?",
        "expected_source": [
        "employee_handbook.txt",
        "confluence_917505.txt",],
        "expected_fact": "30 days",
    },
    {
        "question": "Can employees work remotely?",
        "expected_source": "company_policy.txt",
        "expected_fact": "three days per week",
    },
    {
        "question": "What is the weather today?",
        "expected_source": None,
        "expected_fact": None,
    },
    {
        "question": "An employee has taken 12 vacation days. How many vacation days do they have left?",
        "expected_source": [
        "employee_handbook.txt",
        "confluence_917505.txt",
],
        "expected_fact": "18",
        "threshold": 0.8,
    },
]

def validate_sources(actual_sources, expected_sources):
    if not actual_sources:
        return False

    return all(
        source in expected_sources
        for source in actual_sources
    )

passed_tests = 0

for test in test_cases:
    question = test["question"]
    expected_source = test["expected_source"]
    expected_fact = test["expected_fact"]

    results = retrieve_documents(
        collection,
        question,
        threshold=test.get("threshold", 0.7),
    )

    if results:
        actual_answer, actual_source = generate_answer(question, results)
    else:
        actual_answer = None
        actual_source = None

    if expected_source is None:
        source_passed = actual_source is None or actual_source == []
    else:
        expected_sources = (
            expected_source
            if isinstance(expected_source, list)
            else [expected_source]
        )

        if isinstance(actual_source, list):
            actual_sources = [
                source["document"] for source in actual_source
            ]
        elif actual_source is None:
            actual_sources = []
        else:
            actual_sources = [actual_source]

        source_passed = validate_sources(
            actual_sources,
            expected_sources,
        )

    if expected_fact is None:
        answer_passed = actual_answer is None
    else:
        answer_text = actual_answer.lower() if actual_answer else ""

        if expected_fact == "30 days":
            answer_passed = "30" in answer_text and "vacation days" in answer_text
        else:
            answer_passed = expected_fact.lower() in answer_text

    passed = source_passed and answer_passed

    if passed:
        passed_tests += 1

    print("\nQuestion:", question)
    print("Expected source:", expected_source)
    print("Actual source:", actual_source)
    print("Answer:", actual_answer)
    print("Source:", "PASS" if source_passed else "FAIL")
    print("Answer:", "PASS" if answer_passed else "FAIL")
    print("Result:", "PASS" if passed else "FAIL")

print(f"\nPassed: {passed_tests}/{len(test_cases)}")

print("\n--- Source Validation Test ---")

expected_sources = ["company_policy.txt"]

valid_sources = ["company_policy.txt"]
invalid_sources = [
    "company_policy.txt",
    "employee_handbook.txt",
]

valid_test_passed = validate_sources(
    valid_sources,
    expected_sources,
)

invalid_test_passed = not validate_sources(
    invalid_sources,
    expected_sources,
)

print(
    "Valid sources:",
    "PASS" if valid_test_passed else "FAIL",
)

print(
    "Unexpected source rejected:",
    "PASS" if invalid_test_passed else "FAIL",
)
question = "Can employees work remotely?"
expected_fact = "three days per week"

def evaluate_agent_test(question, expected_fact):
    print("\n--- Agent knowledge Test ---")

    result = run_agent(question)
    answer = result["answer"]

    passed = expected_fact.lower() in answer.lower()

    print("Question:", question)
    print("Expected:", expected_fact)
    print("Actual:", answer)
    print("Result:", "PASS" if passed else "FAIL")
    return passed

total_agent_tests = 0
agent_passed = 0


total_agent_tests += 1
if evaluate_agent_test("Can employees work remotely?", "three days per week"):
    agent_passed += 1

total_agent_tests += 1
if evaluate_agent_test("What is 25 + 17?", "42"):
    agent_passed += 1
total_agent_tests += 1
if evaluate_agent_test(
    "An employee has already taken 12 vacation days. How many vacation days do they have left?",
    "18",
):
    agent_passed += 1


def evaluate_missing_information_test(question):
    print("\n--- Missing Information Test ---")
    result = run_agent(question)
    answer = result["answer"]
    missing_indicators = [
        "couldn't find",
        "no information",
        "not found",
        "doesn't contain",
        "don't have information",
        "wasn't able to find",
        "unable to find information",
    ]
    passed = any(phrase in answer.lower() for phrase in missing_indicators)
    print("Question:", question)
    print("Actual:", answer)
    print("Result:", "PASS" if passed else "FAIL")
    return passed


total_agent_tests += 1
if evaluate_missing_information_test(
    "What is the company's policy on pets in the office?"
):
    agent_passed += 1
print(f"\nAgent tests passed: {agent_passed}/{total_agent_tests}")
