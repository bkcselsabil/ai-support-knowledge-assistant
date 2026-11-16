# AI Support Knowledge Assistant

**A multi-tenant AI-powered knowledge assistant built with Python, Claude, LangGraph, RAG, FastAPI, and PostgreSQL.**

This project is a SaaS-oriented AI application that helps organizations answer employee and support questions using their internal knowledge bases.

It combines **Retrieval-Augmented Generation (RAG)**, semantic vector search, and LLM-powered tool selection to generate context-aware answers grounded in company documents, with source references.

### Key capabilities

- **AI-powered knowledge retrieval:** Search internal documents using Sentence Transformers and ChromaDB.
- **Agentic AI workflows:** Use Claude and LangGraph to route questions and execute appropriate tools.
- **Multi-tenant architecture:** Separate organizations' knowledge bases and conversation data.
- **Authentication:** Integrate Clerk for user authentication and organization context.
- **Persistent conversations:** Store chat history and source references using PostgreSQL and SQLAlchemy.
- **Backend API:** Provide authenticated chat functionality through FastAPI.
- **Containerized development:** Support Docker-based local execution and Alembic database migrations.
- **Automated testing:** Validate retrieval, routing, API behavior, and tenant isolation using pytest.

### Project objective

The goal is to demonstrate practical AI engineering skills by building an end-to-end application that integrates LLMs, vector databases, agent workflows, backend development, authentication, and persistent storage.

The project is intended as a portfolio demonstration and is not yet a production-ready SaaS service.

## Features

### AI and RAG
- **Retrieval-Augmented Generation (RAG):** Retrieve relevant document chunks to ground AI-generated answers.
- **Semantic search:** Generate embeddings using Sentence Transformers and search ChromaDB.
- **Document ingestion:** Process local documents and support Confluence knowledge ingestion.
- **Metadata filtering:** Filter retrieved documents by name, category, and organization.
- **Claude integration:** Generate responses using Anthropic Claude.
- **LangGraph workflows:** Route questions between knowledge retrieval and calculator functionality.
- **Source attribution:** Return document references alongside generated answers.

### Authentication and Multi-Tenancy
- **Clerk authentication:** Authenticate users using session tokens.
- **Organization isolation:** Restrict knowledge retrieval and conversations to the authenticated organization.
- **Conversation access control:** Prevent users from accessing conversations belonging to other users or organizations.

### Backend and Database
- **FastAPI REST API:** Provide authenticated chat endpoints.
- **PostgreSQL persistence:** Store conversations, messages, and source references.
- **SQLAlchemy ORM:** Manage database models and queries.
- **Alembic migrations:** Version and manage database schema changes.

### Development and Testing
- **Docker:** Support containerized backend development.
- **Environment configuration:** Keep local credentials outside version control.
- **Automated tests:** Test API behavior, RAG retrieval, agent routing, database-related behavior, and tenant isolation using pytest.
- **Web chat interface:** Provide a browser-based interface for interacting with the assistant.

## Architecture

The application combines a FastAPI backend, Clerk authentication, LangGraph-based AI workflows, ChromaDB vector retrieval, and PostgreSQL persistence.

### 1. Application architecture

```text
                  User
                   |
                   v
           Web Chat Interface
                   |
                   v
          Clerk Authentication
                   |
                   v
             FastAPI API
                   |
          User + Organization
             Validation
                   |
                   v
           LangGraph Workflow
                   |
          +--------+--------+
          |                 |
          v                 v
     RAG Retrieval     Calculator Tool
          |
          v
       ChromaDB
   (Tenant-filtered)
          |
          v
    Retrieved Context
          |
          v
       Claude LLM
          |
          v
    Answer + Sources
          |
          v
        FastAPI
          |
          +----> PostgreSQL
          |      Conversations
          |      Messages
          |      Source References
          |
          v
      Web Interface
```

### 2. Knowledge ingestion pipeline

```text
Local Documents / Confluence
              |
              v
      Document Processing
              |
              v
          Chunking
              |
              v
     Sentence Transformers
          Embeddings
              |
              v
       Tenant Metadata
              |
              v
           ChromaDB
              |
              v
     Tenant-Filtered Search
```

Each indexed document chunk includes organization-specific metadata, allowing retrieval to be restricted to the authenticated tenant.

### 3. Multi-tenant data isolation

The application uses the authenticated organization identifier to scope knowledge retrieval and conversation access.

- Clerk provides authenticated user and organization context.
- FastAPI validates access to conversations.
- ChromaDB retrieval filters documents by tenant identifier.
- PostgreSQL stores conversation ownership information.

These controls provide application-level tenant isolation. Additional security review and deployment hardening would be required before production use.

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

## Tech Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| LLM | Anthropic Claude |
| AI orchestration | LangGraph, LangChain |
| RAG and embeddings | Sentence Transformers, ChromaDB |
| Backend | FastAPI, Pydantic |
| Authentication | Clerk |
| Database | PostgreSQL, SQLAlchemy |
| Database migrations | Alembic |
| Frontend | HTML, CSS, JavaScript |
| Containerization | Docker, Docker Compose |
| Testing | pytest |
| Knowledge integration | Confluence REST API |
| Configuration | python-dotenv |
| Version control | Git, GitHub |

### Engineering concepts demonstrated

- Retrieval-Augmented Generation (RAG)
- LLM tool calling and workflow orchestration
- Semantic search and vector databases
- Multi-tenant data isolation
- Token-based authentication and authorization
- REST API development
- Relational database design and persistence
- Database schema migrations
- Containerized local development
- Automated testing and API mocking

## Example

**Question**

How many vacation days do employees receive each year?

**Example answer**

Employees receive 30 days of annual vacation. Vacation requests must be submitted through HR.

**Source**

`employee_handbook.txt`

*This example illustrates the expected behavior using the project's sample knowledge base. Actual answers depend on the documents available to the application.*
## Project Structure

```text
ai-support-knowledge-assistant/
│
├── app/
│   ├── main.py                  # FastAPI endpoints
│   └── auth.py                  # Clerk authentication
│
├── db/
│   ├── database.py              # PostgreSQL connection
│   └── models.py                # SQLAlchemy models
│
├── migrations/
│   ├── env.py                   # Alembic configuration
│   └── versions/                # Database migrations
│
├── frontend/
│   ├── index.html               # Chat interface
│   ├── script.js                # Frontend logic
│   └── style.css                # Interface styling
│
├── documents/                   # Sample knowledge documents
├── tests/                       # Automated tests
├── examples/                    # AI experimentation scripts
│
├── agent.py                     # Claude agent and tools
├── rag.py                       # RAG and ChromaDB retrieval
├── confluence_ingestion.py      # Confluence integration
├── evaluate.py                  # AI evaluation script
├── evaluation_dataset.json      # Evaluation examples
│
├── compose.yaml                 # Docker Compose configuration
├── Dockerfile                   # Backend container definition
├── .dockerignore                # Docker build exclusions
├── alembic.ini                  # Alembic configuration
├── create_docker_env.py         # Local Docker environment setup
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git exclusions
└── README.md                    # Project documentation
```

The structure separates API functionality, authentication, database persistence, AI retrieval, and testing into distinct components.

## Prerequisites

To run the AI Support Knowledge Assistant locally, you will need:

### Required software
- **Python 3.11+** — Python runtime (the project was developed using Python 3.13).
- **Git** — Clone and manage the repository.
- **PostgreSQL** — Store conversations, messages, and source references.
- **Docker Desktop** — Run the containerized backend (optional if running Python directly).

### External services
- **Anthropic API key** — Access Claude for AI-generated responses and tool selection.
- **Clerk account** — Configure user authentication and organization-based access.

### Optional integrations
- **Atlassian Confluence** — Connect an external knowledge base using the Confluence REST API.

### Local configuration

The application requires environment variables for the LLM provider, database connection, and authentication service.

Credentials must be stored in a local `.env` file and must never be committed to Git.

## Setup

The following instructions describe how to run the application locally on Windows using PowerShell.

### 1. Clone the repository

```powershell
git clone https://github.com/bkcselsabil/ai-support-knowledge-assistant.git
cd ai-support-knowledge-assistant
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```
### 4. Configure environment variables

Copy the example configuration file to create your local `.env` file:

```powershell
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder values with your own credentials.

The application uses the following environment variables:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic Claude API access |
| `CLERK_SECRET_KEY` | Clerk backend authentication |
| `CLERK_AUTHORIZED_PARTIES` | Allowed frontend origins for Clerk token verification |
| `CLERK_TEST_ORGANIZATION_ID` | Organization identifier used by the authentication configuration |
| `DB_USER` | PostgreSQL username |
| `DB_PASSWORD` | PostgreSQL password |
| `DB_HOST` | PostgreSQL hostname (defaults to `127.0.0.1`) |
| `DB_PORT` | PostgreSQL port (defaults to `5433`) |
| `DB_NAME` | PostgreSQL database name |

Optional Confluence integration requires `CONFLUENCE_BASE_URL`, `CONFLUENCE_EMAIL`, and `CONFLUENCE_API_TOKEN`.

**Security:** Never commit `.env`, API keys, passwords, or private company documents. The `.env.example` file contains placeholders only.

### 5. Initialize the PostgreSQL database

Make sure PostgreSQL is installed and running.

Create an empty PostgreSQL database using the name specified by `DB_NAME` in your `.env` file.

Ensure the configured database user has permission to connect to the database and create tables.

The database connection is configured using:

- `DB_USER`
- `DB_PASSWORD`
- `DB_HOST`
- `DB_PORT`
- `DB_NAME`

The application uses SQLAlchemy to manage database connections and models.

### 6. Apply database migrations

The project uses Alembic to manage database schema changes.

From the project root, run:

```powershell
python -m alembic upgrade head
```

This applies all pending migrations, including the initial schema for conversations, messages, and message source references.

To verify the current migration version:

```powershell
python -m alembic current
```

The database is now ready for the FastAPI application.

### 7. Start the FastAPI backend

Make sure PostgreSQL is running, your `.env` file is configured, and the database migrations have been applied.

From the project root, start the FastAPI development server:

```powershell
python -m uvicorn app.main:app --reload
```

The API will be available at:

- `http://127.0.0.1:8000`
- API documentation: `http://127.0.0.1:8000/docs`

### Authentication

The chat endpoint requires a valid Clerk session token.

Users must authenticate through Clerk and have an active organization context. The backend uses the authenticated user and organization identifiers to enforce conversation access controls and tenant-scoped knowledge retrieval.

Unauthenticated requests or requests without the required organization context are rejected.

**Note:** The development server is intended for local use. Additional configuration and security hardening are required before public deployment.

### 8. Start the frontend

Open a second PowerShell terminal while keeping the FastAPI backend running.

From the project root, run:

```powershell
cd frontend
python -m http.server 5500
```

Open the application in your browser:

`http://127.0.0.1:5500`

### Sign in with Clerk

The frontend integrates Clerk authentication.

1. Sign in using your configured Clerk application.
2. Select an organization associated with your account.
3. Submit a question through the chat interface.

The frontend sends the Clerk session token to the FastAPI backend in the `Authorization` header.

The backend validates the token and uses the authenticated organization context to scope conversation access and document retrieval.

### Local development checklist

Before using the chat interface, ensure that:

- PostgreSQL is running and migrations are applied.
- The FastAPI backend is running on port `8000`.
- The frontend is running on port `5500`.
- Clerk is configured for your local frontend origin.
- Your account has an active organization.
- Your knowledge base contains documents indexed for that organization.

The application is now ready for local testing.

## Testing

The project uses **pytest** to validate AI workflows, retrieval behavior, API functionality, and multi-tenant security.

### Run automated tests

From the project root:

```powershell
python -m pytest -q
```

**Latest verified local result: 59 tests passed.**

### Test coverage areas

- RAG document retrieval and relevance filtering
- ChromaDB metadata and tenant-scoped retrieval
- Claude agent tool execution
- LangGraph workflow routing
- FastAPI endpoint behavior
- Clerk authentication and organization validation
- Conversation ownership and cross-tenant access restrictions
- Source attribution and fallback behavior

### Measure code coverage

If `pytest-cov` is installed, run:

```powershell
python -m pytest --cov=app --cov=rag --cov=agent --cov-report=term-missing
```

Coverage percentages should be reported only after running the coverage command against the current implementation.

Automated tests help validate expected behavior but do not replace security reviews, integration testing, or evaluation of AI-generated answers.

## Evaluation

The project includes an evaluation dataset (`evaluation_dataset.json`) and an evaluation script (`evaluate.py`) to support testing the AI assistant's responses.

### Run evaluation

From the project root:

```powershell
python evaluate.py
```

Evaluation requires the appropriate API configuration and any necessary knowledge-base data.

### Evaluation objectives

The evaluation process is intended to assess:

- **Answer correctness:** Whether generated answers match expected information.
- **Retrieval relevance:** Whether retrieved documents contain information relevant to the question.
- **Groundedness:** Whether answers are supported by retrieved documents.
- **Fallback behavior:** Whether the assistant handles questions that cannot be answered from the knowledge base.

### Future improvements

Planned improvements include automated evaluation metrics, larger evaluation datasets, and regression testing to measure changes in response quality.

**Note:** Automated unit tests and code coverage do not directly measure the factual accuracy of AI-generated answers.

## Current Limitations and Future Improvements

The AI Support Knowledge Assistant is a portfolio project demonstrating an end-to-end, multi-tenant AI application. It is currently intended for local development and testing rather than production deployment.

### Current limitations

- **AI evaluation:** Retrieval relevance, factual accuracy, and hallucination rates require more systematic evaluation.
- **Security:** Authentication and tenant-isolation controls are implemented, but a comprehensive security audit has not been performed.
- **Deployment:** Docker supports local development, but production infrastructure and deployment automation are not configured.
- **Observability:** Centralized logging, monitoring, and performance metrics need improvement.
- **Knowledge ingestion:** Document synchronization, updates, and duplicate handling need further development.

### Planned improvements

- Build an automated RAG evaluation pipeline with measurable quality metrics.
- Add CI workflows using GitHub Actions.
- Improve error handling, logging, and monitoring.
- Implement incremental document ingestion and synchronization.
- Add end-to-end tests for authentication and external integrations.
- Prepare a secure production deployment architecture.
- Add screenshots and a short demonstration of the chat interface.

These improvements are planned and are not presented as completed functionality.

## License

No license has been selected yet. Until a license is added, do not describe the repository as open-source software with an explicit reuse license.