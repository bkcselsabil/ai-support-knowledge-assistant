
from rag import retrieve_documents, collection





test_cases = [
    {
        "question": "How many vacation days do employees get?",
        "expected_source": "employee_handbook.txt",
    },
    {
        "question": "Can employees work remotely?",
        "expected_source": "company_policy.txt",
    },
    {
        "question": "What is the weather today?",
        "expected_source": None,
    },
]

passed_tests = 0
for test in test_cases:
    question = test["question"]
    expected_source= test["expected_source"]
    
    results = retrieve_documents(collection, question)
    
    if results:
        actual_source = results[0]["metadata"]["document"]
    else:
        actual_source = None
    passed = actual_source == expected_source
    if passed: 
        passed_tests +=1
    
    print("\nQuestion:", question)
    print("Expected:", expected_source)
    print("Actual:", actual_source)
    print("Result:", "PASS" if passed else "FAIL")
print(f"\nPassed: {passed_tests}/{len(test_cases)}")