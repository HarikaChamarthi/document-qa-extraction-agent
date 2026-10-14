
# Debugging Report - Document Q&A and Extraction Agent

## 1. Project Overview

This project provides APIs for document upload, RAG-based question answering, structured field extraction, extraction validation, and agentic extraction with retries and human-review handling.

## 2. Environment

- Framework: FastAPI
- Language: Python
- API testing: Swagger UI and pytest
- LLM provider: Google Gemini
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Vector database: ChromaDB
- Logging format: JSON Lines (JSONL)

## 3. Automated Test Results

**Latest test command:** `python -m pytest tests -v`

- Tests passed: 26
- Tests skipped: 2
- Tests failed: 0
- Warnings: 1 Starlette deprecation warning

The two skipped tests are live Gemini evaluation tests that require explicit opt-in. They were not verified by this test run.

### Verified test areas

| Test area | Result |
|---|---|
| Root and health endpoints | Passed |
| Document upload and input validation | Passed |
| Extraction validation | Passed |
| Agentic extraction and retry behavior | Passed |
| Human-review handling | Passed |
| Gemini retry and error-handling logic | Passed |
| RAG prompt security checks | Passed |
| Request ID middleware | Passed |
| Prompt evaluation | Passed |

## 4. Manual Testing

### Document Upload

A text document was uploaded successfully. The API returned HTTP 200 and confirmed that the document was parsed, embedded, and stored in ChromaDB.

### RAG Question Answering

The `/ask-document` endpoint retrieved relevant document content and returned an answer with source information.

### Prompt-Injection Test

A document containing employee information and malicious instructions was uploaded. The tested question returned the factual employee information with a source reference and did not reveal the requested hidden instructions or API key.

This result applies to the tested example and does not guarantee protection against every possible prompt-injection attack.

### LLM Event Logging

The `llm_logs.jsonl` file exists and contains both `rag_retrieval` and `generate_content` events.

Recorded information includes timestamps, request IDs, model names, prompt versions, operation names, latency, success status, and applicable error or retrieval details.

## 5. Issues Observed

### Issue 1: Gemini API failures

**Observation:** Some earlier generation events recorded `success: false` and `error_type: ClientError`.

**Possible cause:** Earlier testing encountered Gemini API quota exhaustion. The available logs do not confirm the cause of every recorded failure.

**Action:** Check the provider's quota and error details if failures recur. Avoid unnecessary repeated live API calls.

### Issue 2: Extraction validation failures

**Observation:** Some log entries contain `extraction_validation` events with `success: false`.

**Meaning:** Those extraction attempts did not satisfy the required-field validation rules.

**Action:** Review the corresponding missing fields and validation errors when investigating an extraction failure.

## 6. Remaining Verification

- **Agentic extraction response:** Verified successfully through Swagger UI. The endpoint returned HTTP 200, extracted all three requested employee fields, populated source references, returned an empty validation error list, and reported `human_review_required: false`.
- **Confidence calculation:** The current implementation calculates confidence from the proportion of requested fields that are populated. This is a completeness score, not an independent measure of factual accuracy.
- **Retry behavior:** The default configuration is set to one retry after the initial extraction attempt. Automated tests for retry behavior and human-review handling passed.
- **Live Gemini evaluation:** Two live Gemini tests remain skipped because they require explicit opt-in. They have not been verified by the latest test run.
- **Repository review:** Review for unintended secrets, generated databases, and local log files.
- **GitHub synchronization:** Confirm that the latest code changes and this debugging report are committed and pushed to GitHub.
- **Production-readiness review:** Confirm the remaining requirements in the task PDF, including configuration, logging, file restrictions, exception handling, and prompt-injection testing.



## 7. Conclusion

The latest automated regression run completed with 26 passed, 2 skipped, and 0 failed. Manual tests also confirmed successful document upload, RAG question answering, prompt-injection handling for the tested example, and LLM event logging.

The project has passed the current automated test suite. The remaining items above require separate verification before all task requirements can be declared complete.
