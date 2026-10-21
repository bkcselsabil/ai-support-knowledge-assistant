# AI Support Knowledge Assistant

An AI-powered support assistant that answers company-policy and employee-support questions using internal documents and Confluence pages. The project combines Retrieval-Augmented Generation (RAG), vector search, and tool-using agents to provide grounded answers with source references.

> **Project status:** RAG, Confluence ingestion, agent tools, and evaluation are implemented. FastAPI and the user interface are planned next.

## Features

- **Document ingestion:** Load local knowledge-base documents and retrieve content from Confluence.
- **Text chunking:** Split documents into smaller pieces for targeted retrieval.
- **Vector search:** Store and search document chunks using ChromaDB.
- **Metadata filtering:** Filter retrieved content by category or document.
- **LLM-powered answers:** Use Anthropic Claude to generate answers from retrieved context.
- **Source references:** Return document names and, for Confluence pages, source URLs.
- **Agent tools:** Support knowledge retrieval and calculations such as remaining vacation days.
- **Evaluation:** Run tests for retrieval, answer correctness, source selection, tool use, and missing information.

## Architecture

### Knowledge ingestion

```text
Local documents ───────┐
                       ├──> Chunking ──> ChromaDB
Confluence pages ──────┘
```

### Question answering

```text
User question
     |
     v
Question routing / agent
     |
     v
Retrieve relevant document chunks
     |
     v
Claude or an appropriate tool
     |
     v
Answer + source references
```

The exact execution path depends on the question and the routing logic.

## Technologies

- Python
- Anthropic Claude API
- ChromaDB
- LangChain and LangGraph
- Confluence REST API
- `requests`
- `python-dotenv`
- `pytest`

## Example

**Question:** Can employees work remotely?

**Answer:** Yes, employees can work remotely up to three days per week.

**Source:** `company_policy.txt`

For Confluence-backed questions, the assistant can also return the relevant page title and URL.

## Project structure

```text
.
├── documents/                   # Local knowledge-base documents
├── rag.py                       # Chunking, retrieval, and answer generation
├── confluence_ingestion.py      # Confluence content ingestion
├── agent.py                     # Agent and tool execution
├── evaluate.py                  # RAG and agent evaluations
├── test_rag.py                  # RAG tests
├── test_langgraph.py            # LangGraph tests
├── test_confluence_connection.py
├── test_confluence_retrieval.py
├── test_combined_documents.py
├── test_chroma_metadata.py
├── requirements.txt
├── .gitignore
└── README.md
```

This lists the main project files; additional experiments and test files may also be present.

## Setup

1. Create and activate a Python virtual environment.
2. Install the dependencies from `requirements.txt`.
3. Configure the required API credentials and Confluence settings in a local `.env` file.
4. Run the assistant using the project's current entry point.

See [Setup Guide](docs/setup.md) for detailed instructions.

**Security:** Never commit `.env`, API tokens, or private company documents to GitHub.

## Testing and evaluation

Run the automated tests:

```powershell
python -m pytest
```

Run the evaluation script:

```powershell
python evaluate.py
```

The latest recorded results before the next development phase were:

- Pytest: 27 tests passed.
- RAG evaluations: 4/4 passed.
- Agent evaluations: 4/4 passed.

Results can change as the code and test cases evolve.

## Roadmap

- [x] Local-document RAG
- [x] Confluence ingestion and retrieval
- [x] Source references and metadata
- [x] Agent tools and calculations
- [x] Initial evaluation suite
- [ ] FastAPI backend
- [ ] Chat interface
- [ ] Improved persistence and tenant isolation
- [ ] Docker and deployment

## License

No license has been selected yet. Add a license before presenting the project as open source.