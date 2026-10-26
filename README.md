# AI Support Knowledge Assistant

An AI-powered support assistant that answers company-policy and employee-support questions using internal documents. Built with Python, Anthropic Claude, ChromaDB, RAG, and FastAPI, it provides answers grounded in retrieved information and returns source references.

The project demonstrates an end-to-end AI application, from document retrieval and tool calling to an API, chat interface, and automated testing.

## Features

- **Retrieval-Augmented Generation (RAG):** Retrieve relevant document chunks before generating answers.
- **Vector search:** Store and search document embeddings using ChromaDB.
- **Document metadata:** Associate retrieved content with document names, categories, and source information.
- **Relevance filtering:** Apply a similarity-distance threshold to help reject irrelevant results.
- **Claude integration:** Generate answers using the Anthropic API.
- **Tool calling:** Support knowledge-base searches and mathematical operations.
- **Source references:** Return source information alongside answers when available.
- **FastAPI backend:** Expose the assistant through HTTP endpoints.
- **Web chat interface:** Submit questions, view answers, and inspect their sources.
- **Automated tests:** Test API behavior, retrieval, agent tools, routing, and fallback behavior.

## Architecture

### Knowledge ingestion and retrieval

```text
Local documents / Confluence pages
                |
                v
        Document processing
                |
                v
            Chunking
                |
                v
           Embeddings
                |
                v
             ChromaDB
                |
                v
      Relevant document chunks
```

### Question answering

```text
User
 |
 v
Web chat interface
 |
 v
FastAPI backend
 |
 v
AI assistant / agent
 |
 +------> Knowledge retrieval
 |              |
 |              v
 |           ChromaDB
 |
 +------> Available tools
 |
 v
Claude response
 |
 v
Answer + source references
```

The exact execution path depends on the question and the relevant application logic. The repository also contains separate LangGraph experiments.

## Technologies

- Python
- Anthropic Claude API
- ChromaDB
- Sentence Transformers
- FastAPI
- HTML, CSS, and JavaScript
- LangChain and LangGraph experiments
- pytest
- python-dotenv
- Confluence REST API integration

## Example

**Question**

How many vacation days do employees receive each year?

**Example answer**

Employees receive 30 days of annual vacation. Vacation requests must be submitted through HR.

**Source**

`employee_handbook.txt`

*This example illustrates the expected behavior using the project's sample knowledge base. Actual answers depend on the documents available to the application.*

## Project structure

```text
.
├── app/
│   └── main.py                 # FastAPI application
├── frontend/
│   ├── index.html              # Chat interface
│   ├── script.js               # Frontend behavior and API requests
│   └── style.css               # Interface styling
├── documents/                  # Local knowledge-base documents
├── tests/
│   ├── test_api.py
│   ├── test_agent.py
│   ├── test_basics.py
│   ├── test_langgraph.py
│   └── test_rag.py
├── agent.py                    # Claude agent and tool execution
├── rag.py                      # Document retrieval and RAG logic
├── confluence_ingestion.py     # Confluence ingestion, if configured
├── evaluation_dataset.json     # Evaluation examples
├── evaluate.py                 # Evaluation script, if configured
├── requirements.txt
├── .gitignore
└── README.md
```

This shows the main application files; additional scripts and tests may exist in the repository.

## Prerequisites

- Python installed on your system
- An Anthropic API key
- Git
- Access to Confluence and its credentials if you want to use Confluence ingestion

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/bkcselsabil/ai-support-knowledge-assistant.git
cd ai-support-knowledge-assistant
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root. Add the variables required by your application, including your Anthropic API key.

For example:

```dotenv
ANTHROPIC_API_KEY=your_anthropic_api_key
```

If you enable Confluence integration, add the additional credentials and configuration expected by your ingestion code.

**Security:** Never commit `.env` files, API keys, access tokens, or private company documents. Use placeholder values in any example configuration file.

### 5. Start the backend

From the project root:

```powershell
uvicorn app.main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Start the frontend

Open a second PowerShell terminal:

```powershell
cd frontend
python -m http.server 5500
```

Open the chat interface at:

```text
http://localhost:5500
```

Keep the backend running while using the frontend.

## Testing

Run the complete test suite from the project root:

```powershell
python -m pytest
```

To view test coverage:

```powershell
python -m pytest --cov=app --cov=rag --cov=agent --cov-report=term-missing
```

Latest recorded results:

- **43 tests passed**
- **92% overall statement coverage**
- `app/main.py`: 100% coverage
- `rag.py`: 95% coverage
- `agent.py`: 87% coverage

These figures describe the latest recorded run and may change as the code evolves.

## Evaluation

The project includes an evaluation dataset and evaluation-related scripts. To run an evaluation, use the command appropriate to the current implementation:

```powershell
python evaluate.py
```

Evaluation results depend on the dataset, implementation, and API configuration. Unit-test coverage alone does not establish answer quality, so retrieval relevance and answer correctness should also be evaluated.

## Current limitations and future improvements

- Improve evaluation of retrieval relevance and factual answer quality.
- Add more tests for external API failures and unusual inputs.
- Improve logging, configuration validation, and error handling.
- Add Docker support and a repeatable deployment workflow.
- Evaluate authentication and access controls before exposing the API publicly.
- Improve persistence and isolation if the application is extended to multiple organizations.
- Add screenshots and a demonstration of the chat interface.

## License

No license has been selected yet. Until a license is added, do not describe the repository as open-source software with an explicit reuse license.