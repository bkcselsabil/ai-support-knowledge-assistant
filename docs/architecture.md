# System Architecture

## Overview

The AI Support Knowledge Assistant combines Retrieval-Augmented Generation (RAG), a vector database, Confluence ingestion, and an agent workflow to answer questions about company policies and employee support.

The system has two main responsibilities:

1. **Knowledge ingestion:** Prepare local documents and Confluence content for retrieval.
2. **Question answering:** Retrieve relevant information, generate answers, and use tools when needed.

## 1. Knowledge ingestion pipeline

```text
Local documents                 Confluence pages
      |                                |
      v                                v
Load and extract text          Fetch and filter pages
      |                                |
      v                                v
Prepare document chunks        Prepare page chunks
      |                                |
      +---------------+----------------+
                      |
                      v
             Combine all chunks
                      |
                      v
                   ChromaDB
```

### Local documents

The application loads text files from the `documents/` directory. The text is split into smaller chunks so retrieval can identify the sections most relevant to a question.

### Confluence ingestion

The Confluence connector uses the Confluence REST API to retrieve pages from a configured space. It filters out irrelevant pages and prepares the remaining content for retrieval.

Confluence chunks include metadata such as:

- Source type
- Page title
- Page ID
- Space key
- Page URL
- Document identifier

### Chunk storage

Local and Confluence chunks are combined and stored in ChromaDB. Metadata helps the application identify where each chunk originated and supports filtering and source attribution.

## 2. RAG question-answering pipeline

```text
User question
      |
      v
Question classification
      |
      v
Retrieve relevant chunks
      |
      v
Filter results by relevance
      |
      v
Build context for Claude
      |
      v
Generate answer
      |
      v
Return answer and source references
```

### Retrieval

The application converts the question into a vector representation through ChromaDB's configured embedding function and searches for similar document chunks.

Retrieval can apply metadata filters, such as a document category, and a distance threshold to reject results that are not sufficiently relevant.

A lower distance indicates a closer match in the current ChromaDB configuration.

### Answer generation

Claude receives the question and the retrieved context. It generates an answer based on the available information.

The application also prepares source references from the retrieved metadata. These can include a local filename or a Confluence page title and URL.

## 3. Agent workflow

The agent workflow supports questions that need knowledge retrieval, calculations, or a combination of both.

Examples include:

| Question | Required capability |
|---|---|
| Can employees work remotely? | Knowledge retrieval |
| What is 25 + 17? | Calculator tool |
| An employee has taken 12 vacation days. How many remain? | Retrieve the annual allowance and calculate the remainder |
| What is the company's policy on pets in the office? | Detect missing information and respond appropriately |

The workflow uses routing to select an appropriate path. For questions requiring a calculation based on company policy, the system can retrieve the relevant information and then use a calculation tool.

The agent should not invent company policies when the knowledge base does not contain relevant information.

## 4. Evaluation and testing

The project uses automated tests and evaluation scripts to check system behavior.

The current evaluation approach covers:

- Retrieval of expected source documents
- Presence of expected facts in answers
- Agent tool selection and execution
- Calculation correctness
- Handling of missing information
- Confluence ingestion and metadata
- Retrieval from combined local and Confluence content

Run the automated tests with:

```powershell
python -m pytest
```

Run the RAG and agent evaluations with:

```powershell
python evaluate.py
```

## 5. Current limitations

- The project is currently run locally from Python.
- A FastAPI backend and chat interface have not yet been implemented.
- ChromaDB persistence and collection lifecycle management need further review before production use.
- Authentication, authorization, tenant isolation, and production deployment remain future work.
- Retrieval and answer quality depend on document quality, chunking, metadata, and evaluation coverage.

## 6. Planned architecture

The next development phase will introduce an API layer:

```text
Chat interface
      |
      v
FastAPI backend
      |
      v
Existing agent and RAG services
      |
      +----> ChromaDB
      |
      +----> Claude API
      |
      +----> Confluence API
```

This will allow a user interface to communicate with the existing assistant through HTTP requests without relying on terminal input and output.