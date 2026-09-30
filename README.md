# AI Support Knowledge Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions using information stored in company documents.

## How it works

The application:

1. Loads documents from the `documents/` folder
2. Splits documents into meaningful chunks
3. Stores the chunks in ChromaDB
4. Retrieves the most relevant information for a user's question
5. Sends the retrieved context to Claude
6. Generates an answer based only on the retrieved information
7. Shows the source document

## Architecture

Documents
↓
Chunking
↓
ChromaDB
↓
Similarity Search
↓
Relevant Context
↓
Claude
↓
Answer + Source

## Technologies

- Python
- Anthropic Claude API
- ChromaDB
- python-dotenv

## Example

Question:

> Can employees work remotely?

Answer:

> Yes, employees can work remotely up to three days per week.

Source:

`company_policy.txt`

## Project Structure

```text
.
├── documents/
│   ├── employee_handbook.txt
│   └── company_policy.txt
├── rag.py
├── requirements.txt
├── .gitignore
├── README.md
└── .env