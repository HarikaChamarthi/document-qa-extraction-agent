# Document Q&A and Extraction Agent

An AI-powered document processing application built with FastAPI, Google Gemini, Sentence Transformers, and ChromaDB. The project supports document question answering, structured field extraction, validation, and retry-based extraction workflows.

## Features

- **Document Processing:** Upload documents and extract readable text.
- **Text Chunking:** Divide document text into smaller chunks for retrieval.
- **Semantic Search:** Generate embeddings and retrieve relevant document chunks.
- **RAG-Based Question Answering:** Use retrieved document context to answer questions with Gemini.
- **Structured Extraction:** Extract requested fields from documents.
- **Data Validation:** Identify missing required fields.
- **Agentic Extraction:** Retry extraction when validation fails and flag results requiring human review.
- **Error Handling:** Handle invalid uploads and common processing failures.
- **Logging:** Record API request and LLM operation details.
- **Automated Testing:** Test API behavior and extraction prompts.

## Tech Stack

- Python
- FastAPI and Uvicorn
- Google Gemini API
- Sentence Transformers — `all-MiniLM-L6-v2`
- ChromaDB
- Pydantic
- PyPDF and python-docx
- Pytest

## Architecture

1. **Upload:** Receive a document through the API.
2. **Parse:** Extract readable text from the document.
3. **Chunk:** Split the text into smaller sections.
4. **Embed:** Convert text chunks into numerical vectors.
5. **Store:** Persist vectors and associated document information in ChromaDB.
6. **Retrieve:** Search for relevant chunks based on a question.
7. **Generate:** Use Gemini to answer questions or extract requested fields.
8. **Validate:** Check the extracted data for missing required fields.
9. **Retry:** Attempt extraction again when validation fails.
10. **Review:** Flag results for human review when necessary.

## Project Structure

```text
ai_engineering_practical/
├── app/
│   ├── main.py
│   ├── models/
│   ├── routes/
│   └── services/
│       ├── chunker.py
│       ├── document_parser.py
│       ├── embedding_service.py
│       ├── llm_logger.py
│       ├── semantic_search.py
│       └── vector_store.py
├── tests/
│   ├── test_api.py
│   ├── test_llm_evaluation.py
│   └── test_prompt_evaluation.py
├── main.py
├── test_gemini.py
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.13 or a compatible Python version
- A Google Gemini API key for LLM-powered functionality

### 1. Clone the repository

```bash
git clone https://github.com/HarikaChamarthi/document-qa-extraction-agent.git
cd document-qa-extraction-agent
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install fastapi uvicorn pydantic python-multipart pypdf python-docx python-dotenv numpy sentence-transformers google-genai chromadb pytest httpx
```

### 4. Configure the API key

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your actual API key. Never commit your API key to GitHub.

### 5. Start the application

```powershell
python -m uvicorn app.main:app --reload --port 8001
```

Open the interactive API documentation:

http://127.0.0.1:8001/docs

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/upload-document` | Upload and process a document |
| POST | `/ask-document` | Ask questions about an uploaded document |
| POST | `/extract-fields` | Extract requested fields |
| POST | `/validate-extraction` | Validate extracted data |
| POST | `/agentic-extract` | Run extraction with validation and retries |

Use the Swagger documentation at `/docs` to inspect the request formats and responses.

## Testing

Run the automated test suite:

```powershell
python -m pytest tests -v
```

The regular test suite includes API tests and prompt-evaluation tests. Live Gemini evaluation tests are skipped unless explicitly enabled, because they make real API calls and may consume API quota.

Latest recorded test result: **26 passed, 2 skipped**.

## Data and Security

- `.env` contains the API key and must remain private.
- `chroma_db/` stores persistent vector data locally.
- `llm_logs.jsonl` contains local operation logs.
- `.gitignore` excludes these files and the virtual environment from Git tracking.

Do not upload private documents, API keys, or other sensitive information to a public repository.

## Future Improvements

- Expand extraction evaluation datasets.
- Improve request-level logging and observability.
- Add authentication and authorization.
- Add deployment configuration and production monitoring.

## Author

Harika Chamarthi

GitHub: https://github.com/HarikaChamarthi
