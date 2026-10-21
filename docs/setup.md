# Setup Guide

This guide explains how to configure and run the AI Support Knowledge Assistant locally on Windows.

## Prerequisites

- Python 3.13 or another version supported by the project's dependencies.
- Git.
- An Anthropic API key.
- Confluence credentials and access to the configured Confluence space if you want to use Confluence ingestion.

## 1. Clone the repository

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ai-support-knowledge-assistant
```

Replace the placeholder with the actual repository URL.

If you already have the project locally, open a terminal in the project directory instead.

## 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Configure environment variables

Create a local `.env` file in the project root. Add the variable names expected by the application and fill in your own credentials.

For example, the configuration may include:

```dotenv
ANTHROPIC_API_KEY=your_anthropic_api_key
CONFLUENCE_BASE_URL=https://your-domain.atlassian.net
CONFLUENCE_EMAIL=your_email@example.com
CONFLUENCE_API_TOKEN=your_confluence_api_token
```

Use the exact variable names expected by the current source code. These names are examples of the intended configuration; verify them before running the application.

**Never commit `.env` or real API credentials to GitHub.**

## 5. Run the assistant

Use the entry point currently supported by the project:

```powershell
python rag.py
```

Follow the terminal prompt to ask a question.

## 6. Run the evaluation suite

Run the automated tests:

```powershell
python -m pytest
```

Run the RAG and agent evaluation script:

```powershell
python evaluate.py
```

The evaluation script checks expected sources, answer facts, agent tool behavior, and handling of questions that cannot be answered from the knowledge base.

## Troubleshooting

### Missing environment variables

Check that `.env` exists in the project root and that its variable names match those read by the application.

### Dependency errors

Confirm that `.venv` is activated and install the dependencies again with `pip install -r requirements.txt`.

### Confluence connection errors

Check the base URL, email, API token, access permissions, and availability of the configured Confluence space.

### Missing or stale retrieval results

Check that document ingestion completed successfully and that the expected documents and metadata are present in the ChromaDB collection.