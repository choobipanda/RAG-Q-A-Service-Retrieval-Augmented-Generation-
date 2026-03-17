# Assignment 3 — RAG Q&A Service

A FastAPI service that ingests documents, performs semantic vector search,
and generates grounded answers with citations using OpenAI.

---

## Project Structure

```
rag_service/
├── main.py           # FastAPI app — all three endpoints
├── store.py          # In-memory chunk + session stores
├── chunker.py        # Fixed-size word-based chunking with overlap
├── embeddings.py     # OpenAI text-embedding-3-small integration
├── retrieval.py      # Manual cosine similarity search
├── llm.py            # Grounded prompt construction + GPT-4o-mini call
├── tests.py          # Unit + endpoint + mock LLM tests
├── requirements.txt
└── README.md
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY=sk-...
```

### 3. Start the server

```bash
uvicorn main:app --reload
```

The API will be live at `http://localhost:8000`.
Interactive docs at `http://localhost:8000/docs`.

---

## API Usage

### Ingest a document

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "cs_notes", "text": "Recursion is when a function calls itself..."}'
```

### Search for relevant chunks

```bash
curl "http://localhost:8000/search?query=What+is+recursion&k=3"
```

### Ask a grounded question

```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-session", "question": "What is recursion?", "k": 4}'
```

---

## Run Tests

```bash
pytest tests.py -v
```

The test suite uses mocked OpenAI calls — no API key is needed to run tests.

---

## Reflection Questions

### 1. Why does grounding reduce hallucinations?

When a model generates answers freely from parametric memory, it may
confidently produce plausible-sounding but incorrect facts — this is
hallucination. Grounding forces the model to reason only over a fixed,
verifiable context window. The system prompt explicitly instructs the model
to answer only from the provided passages. Any question whose answer is
absent from the retrieved context triggers a "I don't have enough information"
response rather than an invented one. Citations also enable humans to verify
each claim against its source.

### 2. How do chunk size and overlap affect retrieval quality?

**Chunk size** controls the granularity of retrieval. Smaller chunks (50–100
words) are more precise — they match narrow queries well — but may omit
surrounding context the LLM needs to answer fully. Larger chunks (400–600
words) give the LLM more context per hit but dilute the embedding signal,
making it harder to distinguish relevant from tangentially related passages.

**Overlap** prevents information loss at chunk boundaries. Without overlap,
a sentence split across two chunks might never be retrieved as a unit.
An overlap of ~15–20% of the chunk size (e.g., 50 words for a 300-word chunk)
is a practical starting point; larger overlaps increase storage and embedding
cost with diminishing returns.

### 3. What is the difference between semantic search and keyword search?

**Keyword search** (BM25, TF-IDF) looks for exact or near-exact token matches.
It is fast, interpretable, and works well when queries and documents share
vocabulary. It fails on paraphrases: "car" does not match "automobile."

**Semantic search** encodes queries and documents into a shared vector space
where proximity captures meaning. "What causes fever?" and "elevated body
temperature triggers" become close vectors even with no shared tokens.
The trade-off is higher latency (embedding + ANN search), sensitivity to the
embedding model's training distribution, and harder interpretability.
Hybrid approaches (BM25 + dense retrieval re-ranked) often outperform either alone.

### 4. What are common failure modes of RAG systems?

- **Retrieval failures**: The relevant chunk is not retrieved (wrong k, poor
  chunking at a boundary, or a query phrasing mismatch). The model then answers
  from nothing, either refusing or hallucinating.
- **Context window overflow**: Too many chunks exceed the model's context limit,
  forcing truncation of relevant passages.
- **Irrelevant chunk injection**: A high-scoring but topically unrelated chunk
  confuses the model and degrades answer quality.
- **Multi-hop gaps**: A question that requires combining information from chunks
  that individually score low may never be answered correctly.
- **Stale knowledge**: In-memory stores are not persisted; a restart loses all
  ingested documents.
- **Prompt injection**: A malicious document could instruct the model to ignore
  its system prompt.

### 5. Why are citations important in AI systems?

Citations make AI answers **verifiable**: a user can check the source passage
and assess whether the model's interpretation is accurate. They establish
**accountability** — it is clear which documents the answer is derived from,
which matters in legal, medical, or academic contexts. Citations also signal
**confidence boundaries**: if no high-scoring chunk exists, the absence of
citation is itself informative. Finally, they support **debugging**: a wrong
answer with citations is far easier to trace and fix than a wrong answer with
no provenance.
