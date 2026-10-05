from agent import run_agent
from rag import retrieve_documents, generate_answer, collection

test_cases = [
    {
        "question": "How many vacation days do employees get?",
        "expected_source": "employee_handbook.txt",
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
]
passed_tests = 0

for test in test_cases:
    question = test["question"]
    expected_source = test["expected_source"]
    expected_fact = test["expected_fact"]
    results = retrieve_documents(collection, question)

    if results:
        actual_answer, actual_source = generate_answer(question, results)
        # print("Expected answer:", test["expected_answer"])
        # print("Actual answer:", actual_answer)
    else:
        actual_answer = None
        actual_source = None
    source_passed = actual_source == expected_source

    if expected_fact is None:
        answer_passed = actual_answer is None
    else:
        answer_passed = expected_fact.lower() in actual_answer.lower()
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


agent_passed = 0
print("\n--- Agent Test ---")

question = "What is 25 + 17?"
expected_answer = "42"

answer = run_agent(question)

passed = expected_answer in answer
if passed:
    agent_passed += 1

print("Question:", question)
print("Expected:", expected_answer)
print("Actual:", answer)
print("Result:", "PASS" if passed else "FAIL")
print(f"\nPassed: {passed_tests}/{len(test_cases)}")

question = "Can employees work remotely?"
expected_fact = "three days per week"
def evaluate_agent_test(question, expected_fact):
    print("\n--- Agent knowledge Test ---")


    answer = run_agent(question)

    passed = expected_fact in answer.lower()


    print("Question:", question)
    print("Expected:", expected_fact)
    print("Actual:", answer)
    print("Result:", "PASS" if passed else "FAIL")
    return passed
total_agent_tests = 0
agent_passed = 0


total_agent_tests += 1
if evaluate_agent_test(
    "Can employees work remotely?",
    "three days per week"
):
    agent_passed += 1

total_agent_tests += 1
if evaluate_agent_test(
    "What is 25 + 17?",
    "42"):
    agent_passed += 1
total_agent_tests += 1
if evaluate_agent_test(
    "An employee has already taken 12 vacation days. How many vacation days do they have left?",
    "18"
):
    agent_passed += 1

def evaluate_missing_information_test(question):
    print("\n--- Missing Information Test ---")
    answer = run_agent(question)
    missing_indicators = [
        "couldn't find",
        "no information",
        "not found",
        "doesn't contain",
        "don't have information",
    ]
    passed = any( phrase in answer.lower()
    for phrase in missing_indicators
    )
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