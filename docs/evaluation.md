# Evaluation and Testing

## Purpose

The project uses automated tests and evaluation scripts to verify retrieval quality, answer correctness, source attribution, agent tool use, and handling of missing information.

## 1. Automated tests

Run the test suite with:

```powershell
python -m pytest
```

The latest confirmed result before the documentation work was **27 tests passed**. Rerun the suite to verify the current code after subsequent changes.

The tests cover components such as RAG behavior, metadata handling, LangGraph workflows, and Confluence integration.

## 2. RAG evaluations

Run:

```powershell
python evaluate.py
```

The RAG evaluation cases check whether the system:

- Retrieves an expected source document.
- Produces an answer containing an expected fact.
- Handles questions with no relevant information.
- Correctly answers questions involving vacation allowance.

The latest recorded evaluation result was **4/4 cases passed**.

### Source evaluation

A question can return more than one source. The evaluation checks whether the expected document is included in the returned sources rather than requiring it to be the only source.

For Confluence-backed results, source metadata can include the page title, document identifier, and page URL.

## 3. Agent evaluations

The agent evaluation cases cover:

- Knowledge retrieval for company policy.
- Calculator tool use.
- Combining retrieved policy information with a calculation.
- Handling questions about information missing from the knowledge base.

The latest recorded result was **4/4 agent cases passed**.

## 4. Example evaluation cases

| Test case | Expected behavior |
|---|---|
| Remote-work policy | Retrieve the policy and identify the three-day weekly limit |
| Annual vacation allowance | Identify the 30-day allowance for full-time employees |
| Remaining vacation days | Calculate 30 minus 12 to produce 18 |
| Unknown office pet policy | Explain that the knowledge base does not provide the answer |

## 5. Current limitations

The current evaluation suite is an initial set of targeted checks. Passing these tests does not guarantee correctness for every question.

Future improvements could include:

- More retrieval precision and recall tests.
- Tests for incorrect or conflicting source documents.
- Regression tests for each supported routing path.
- Evaluation of citation correctness and completeness.
- Tests for API failures, malformed inputs, and unavailable external services.
- Broader datasets and repeatable measurements of answer quality.

## 6. Reproducing the results

To reproduce the current checks, activate the project's virtual environment, configure the required environment variables, and run:

```powershell
python -m pytest
python evaluate.py
```

Record new results after meaningful changes rather than assuming previous results still apply.