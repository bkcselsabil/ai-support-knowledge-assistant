
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
    expected_source= test["expected_source"]
    expected_fact = test["expected_fact"] 
    results = retrieve_documents(collection, question)
    
    if results:
        actual_answer, actual_source = generate_answer(
            question,
            results
        )
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
        passed_tests +=1
    

    print("\nQuestion:", question)
    print("Expected source:", expected_source)
    print("Actual source:", actual_source)
    print("Answer:", actual_answer)
    print("Source:", "PASS" if source_passed else "FAIL")
    print("Answer:", "PASS" if answer_passed else "FAIL")
    print("Result:", "PASS" if passed else "FAIL")
print(f"\nPassed: {passed_tests}/{len(test_cases)}")