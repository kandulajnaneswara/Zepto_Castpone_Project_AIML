# Support Assistant Module (`/support_assistant`)

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-used-blue?logo=huggingface)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-1C3C3C)](https://www.langchain.com/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B6B)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)


A fully offline-gradable RAG (Retrieval-Augmented Generation) service
for Zepto customer support: it embeds Zepto's own policy documents, routes
each incoming query through a LangGraph pipeline, retrieves grounded context
when needed, and returns a structured, schema-validated JSON answer via a
FastAPI endpoint.

## Overview

Support Assistant module demonstrates an end-to-end offline-gradable Retrieval-Augmented Generation (RAG) workflow for Zepto customer support.

### Key Technologies

- FastAPI for the API layer
- LangGraph for query routing and workflow orchestration
- ChromaDB for persistent vector retrieval
- Sentence Transformers for local embeddings
- Pydantic for structured response validation
- Docker for containerized execution

### High-Level Flow

```text
Customer Query
      ↓
Intent Classification
      ↓
 ┌────┴──────────────┐
 ↓                   ↓
Policy Question   General Question
 ↓                   ↓
Embedding +       Direct Answer
ChromaDB Top-3
 ↓
Grounded Answer
 └───────┬───────────┘
         ↓
 Pydantic Validation
         ↓
      FastAPI
      POST /ask
```

## Contents

```
support_assistant/
├── docs/                              8 policy documents (doc_01.txt ... doc_08.txt)
├── ingest.py                          Loads, chunks, embeds docs -> stores in ChromaDB
├── schemas.py                         Pydantic request/response models
├── prompt_template.py                 Structured prompt (role-context-task-format-length)
├── graph.py                           LangGraph StateGraph: 3 nodes + conditional edge
├── main.py                            FastAPI app (POST /ask)
├── requirements.txt                   Module dependencies
├── Dockerfile                         Container build for the FastAPI app
├── README.md
└── output_images
      ├── FastAPI_UI.png                     Zepto Support Assistant FastAPI User Interface(UI)
      ├── Localhost_URL_JSON_response.png    Pretty print JSON response of Localhost URL
      ├── query_1.png                        AskRequest and AskResponse of Query 1 in POST
      ├── query_2.png                        AskRequest and AskResponse of Query 2 in POST
      ├── query_3.png                        AskRequest and AskResponse of Query 3 in POST
      └── Uvicorn_terminal_window.png        Uvicorn server process terminal output                         
```

`chromadb/` (the persisted vector store) is created automatically the
first time `ingest.py` runs — it is not committed empty, but is regenerated
by running the ingestion step below.

## MOCK_LLM — read this first

Every LLM call in this module is based on a single environment
variable, `MOCK_LLM`:

- **Unset, or `MOCK_LLM=1` (default):** fully deterministic, rule-based mock
  logic. No signup, no API key, no network call to any LLM provider. **This
  is the graded baseline** — the entire submission is correct using only
  this path.
- **`MOCK_LLM=0` (optional, ungraded extension):** routes generation through
  an actual LLM instead.

## Install / Run Steps

### 1. Install dependencies
```bash
cd support_assistant
pip install -r requirements.txt
```
This installs `sentence-transformers`, which will download the
`all-MiniLM-L6-v2` model from Hugging Face **the first time it runs** — this needs 
internet access once; after that it's cached locally and runs offline.

### 2. Run ingestion (embeds the 8 docs into ChromaDB)
```bash
python ingest.py
```
Expected output ends with:
```
Ingestion complete. Collection now contains 8 embedded chunks.
```
This creates a `chromadb/` folder in `support_assistant/` — the persistent vector store `main.py` reads from.

### 3. Start the API server
```bash
uvicorn main:app --host 127.0.0.1 --port 7861
```
Leave `MOCK_LLM` unset (or `MOCK_LLM=1`) — this is the graded default.

### 4. Call the endpoint
Then open:

`http://127.0.0.1:7861/docs`

Go to POST/ask in the webpage. Use 'Tryitout'. Provide the your question across the "query":
```bash
curl -X POST http://127.0.0.1:7861/ask -H "accept: application/json" -H "Content-Type: application/json" -d "{\"query\": \"string\"}"
```

## Example API Calls (actual output here)

**Example 1 — should trigger retrieval (`policy_question`):**
```bash
curl -X POST http://127.0.0.1:7861/ask -H "accept: application/json" -H "Content-Type: application/json" -d "{\"query\": \"What is your refund policy?\"}"
```
Actual response:
```json
{
  "answer": "Based on retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery...",
  "intent": "policy_question",
  "sources": ["doc_02",
  "doc_06",
  "doc_05"],
  "confidence": 1.0
}
```


**Example 2 — should NOT trigger retrieval (`general_question`):**
```bash
curl -X POST http://127.0.0.1:7861/ask -H "accept: application/json" -H "Content-Type: application/json" -d "{\"query\": \"What is the capital of France?\"}"
```
Actual response:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "intent": "general_question",
  "sources": [],
  "confidence": 1.0
}
```

**Example 3 — should trigger retrieval (`policy_question`):**
```bash
curl -X POST http://127.0.0.1:7861/ask -H "accept: application/json" -H "Content-Type: application/json" -d "{\"query\": \"What is the return window for damaged grocery items?\"}"
```
Actual response:
```json
{
  "answer": "Based on retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery...",
  "intent": "policy_question",
  "sources": ["doc_02",
  "doc_06",
  "doc_05"],
  "confidence": 1.0
}
```

Stop by pressing `CTRL + C`.


## Architecture — RAG Pipeline Walkthrough

**Ingestion** — `ingest.py` reads all 8 `.txt` files from `docs/` via
`load_documents()`. Chunking (`chunk_documents()`) uses one chunk per
document: each policy doc is already short and topically self-contained
(a single paragraph), so per-document chunking is a reasonable, simple
scheme here rather than splitting further.

**Embedding** — `embeddings_and_store()` in `ingest.py` loads
`all-MiniLM-L6-v2` via `sentence-transformers` (local, no API key) and
encodes each chunk into a vector. Vectors are upserted into a persistent
ChromaDB collection named `Zepto_policy_corpus` (stored under `chromadb/`),
with each chunk's `doc_id` (e.g. `doc_01`) used as its Chroma ID — these
same IDs are what later populate the `sources` field in the API response.

**Retrieval** — happens inside the `retrieve_and_answer` node in
`graph.py`, via `retrieve_top_chunks()`. The incoming query is embedded
with the same `all-MiniLM-L6-v2` model, then `collection.query(...)`
retrieves the top-3 most similar chunks from ChromaDB by cosine similarity.
This step runs for real in both `MOCK_LLM` states, since local embedding +
ChromaDB needs no network call or API key.

**Generation** — also inside `retrieve_and_answer` (for policy questions)
and `direct_answer` (for general questions), both in `graph.py`. This is
the one stage that branches on `MOCK_LLM`:
- **Mock (default):** `retrieve_and_answer` returns a canned string —
  `f"Based on the retrieved context: {top_chunk_snippet}"` — built directly
  from the top retrieved chunk's first ~200 characters, no LLM call.
  `direct_answer` returns a fixed string, no LLM call.
- **Optional `MOCK_LLM=0`:** `retrieve_and_answer` would instead build a
  prompt via `prompt_template.build_prompt()` (the structured
  role–context–task–format–length template with the negative constraint and
  few-shot example) and call a real LLM, grounded only in the retrieved
  chunks. `direct_answer` would call the LLM directly, with no retrieval.

**Routing** — `classify_intent` (in `graph.py`) classifies the query first.
In mock mode, it's a keyword heuristic: if the lowercased query contains
any of `delivery, return, refund, membership, tracking, cancel, gift card,
support hours`, it's `policy_question`; otherwise `general_question`. A
conditional edge (`route_after_classification`) then sends
`policy_question` queries to `retrieve_and_answer` and `general_question`
queries to `direct_answer`.

**Data flow, end to end:**
```
query --> classify_intent (keyword heuristic)
              |
              +-- policy_question --> retrieve_and_answer
              |                         (embed query -> ChromaDB top-3 -> canned/LLM answer)
              |
              +-- general_question --> direct_answer
                                        (canned/LLM answer, no retrieval)
                              |
                              v
                  Pydantic AskResponse (answer, sources, confidence)
                              |
                              v
                    FastAPI POST /ask (JSON response)
```

## Structured Output Schema

Enforced via `schemas.AskResponse` (Pydantic): `answer: str`, `intent : Literal["policy_question", "general_question"]`,
`sources: List[str]`, `confidence: float` (0–1). In mock mode this is
populated deterministically in code — `sources` = retrieved chunk IDs (or
`[]` for general questions), `confidence` = a fixed `1.0` — since no LLM
output exists to fail validation. The optional real-LLM path includes
retry logic (`call_actual_llm_with_llm_validation` in `graph.py`) that would
retry up to 2 additional times with a corrective instruction before
returning a clearly marked error response.

## API Quick Reference

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/ask` | Submit a customer-support query and receive a structured JSON response |
| `GET` | `/docs` | Open the FastAPI interactive Swagger UI |

### Request Example

```json
{
  "query": "What is your refund policy?"
}
```

### Response Fields

| Field | Description |
|---|---|
| `answer` | Generated or mock answer |
| `intent` | `policy_question` or `general_question` |
| `sources` | Retrieved document IDs |
| `confidence` | Confidence value between 0 and 1 |

## Docker

```bash
docker build -t zepto-support-assistant .
docker run -p 7861:7860 zepto-support-assistant
```

Open `http://127.0.0.1:7861/docs` and test `POST /ask`. Then Stop it by pressing `CTRL + C`.

The container runs `ingest.py` on startup (regenerating the ChromaDB
collection inside the container), then starts `uvicorn` on port 7860.
`MOCK_LLM` defaults to `1` inside the container — the graded baseline
requires no API key, secret, or network access to any LLM provider.



## Optional Extensions

**optional and ungraded** — The code has placeholder
functions (`classify_intent_with_llm`, `call_actual_llm_with_validation`,
`call_actual_llm_direct` in `graph.py`) that clearly raise
`NotImplementedError` if `MOCK_LLM=0` is set.

1. **Actual LLM via Groq's free tier** (`MOCK_LLM=0`) 
2. **Deployment to Hugging Face Spaces**

---


### Visual outputs

Module output screenshots are shown as below:

- Localhost URL's JSON response
  
  <img width="520" height="218" alt="Localhost_URL_JSON_response" src="https://github.com/user-attachments/assets/b8d006cb-5899-4df8-bcc4-53dce359d681" />

- Support assistant's FastAPI UI
  
  <img width="1500" height="773" alt="FastAPI_UI" src="https://github.com/user-attachments/assets/6df3d576-470f-4fb5-aa8e-0bc3840c3cf7" />

- AskRequest & AskResponse (for query 1, query 2 and query 3)

  **query 1**
  
  <img width="620" height="688" alt="query_1" src="https://github.com/user-attachments/assets/bf04c7ae-dfde-4725-8ef0-86a27f5bac30" />

  **query 2**
  
  <img width="620" height="688" alt="query_2" src="https://github.com/user-attachments/assets/b9f847f6-b4af-4672-b8b8-f9967ceb16cd" />

  **query 3**
  
  <img width="620" height="688" alt="query_3" src="https://github.com/user-attachments/assets/cde6df5a-a39c-421c-a6b4-be74dd1a7c15" />

- Uvicorn server process terminal window output(Output while running the main.py file)
  
  <img width="1001" height="331" alt="Uvicorn_terminal_window" src="https://github.com/user-attachments/assets/0ecc9f6a-8552-48d0-afec-ed9d040905a6" />

