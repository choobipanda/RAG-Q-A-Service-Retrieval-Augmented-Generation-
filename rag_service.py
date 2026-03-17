import math
import os
import uuid

from fastapi import FastAPI, HTTPException, Query
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

chunk_store: dict = {}
session_store: dict = {}

_openai_client = None

def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before running."
            )
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks of approximately `chunk_size` words.
    """
    if not text or not text.strip():
        return []

    words = text.split()

    if len(words) <= chunk_size:
        return [text.strip()]

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return chunks

async def embed_text(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """
    Generate a vector embedding for the given text using OpenAI.
    """
    client = get_openai_client()
    cleaned = text.replace("\n", " ")  # OpenAI recommendation for quality
    response = await client.embeddings.create(input=cleaned, model=model)
    return response.data[0].embedding

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_sim(A, B) = (A · B) / (||A|| * ||B||)
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot_product / (mag_a * mag_b)

def cosine_similarity_search(
    query_embedding: list[float],
    store: dict,
    k: int = 3,
) -> list[tuple[str, float]]:
    """
    Return the top-k chunk IDs most similar to the query embedding.
    """
    scores = [
        (chunk_id, cosine_similarity(query_embedding, data["embedding"]))
        for chunk_id, data in store.items()
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

SYSTEM_PROMPT = """You are a precise question-answering assistant.
You ONLY answer based on the provided context passages below.
If the answer is not found in the context, say:
"I don't have enough information in the provided documents to answer that question."

Do NOT use any outside knowledge. Cite which context passage(s) support your answer
by referencing them as [chunk_id] inline where relevant."""


def build_grounded_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Construct the user message with injected context and the question.
    """
    context_lines = ["CONTEXT:"]
    for chunk in retrieved_chunks:
        context_lines.append(f"[{chunk['chunk_id']}]: {chunk['text']}")
    return "\n".join(context_lines) + f"\n\nQUESTION: {question}"

async def generate_grounded_answer(
    question: str,
    retrieved_chunks: list[dict],
    history: list[dict],
    model: str = "gpt-4o-mini",
) -> str:
    """
    Generate an answer grounded in the retrieved chunks.

    Args:
        question:         The user's question.
        retrieved_chunks: List of {chunk_id, score, text} dicts.
        history:          Prior conversation turns [{role, content}, ...].
        model:            OpenAI chat model to use.

    Returns:
        The assistant's answer string.
    """
    client = get_openai_client()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": build_grounded_prompt(question, retrieved_chunks)})

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,  # Low temperature for factual grounded answers
    )
    return response.choices[0].message.content


class IngestRequest(BaseModel):
    doc_id: str = Field(..., example="intro_cs_notes")
    text: str = Field(..., example="Long document text goes here...")

class IngestResponse(BaseModel):
    doc_id: str
    chunks_added: int

class SearchResult(BaseModel):
    chunk_id: str
    score: float
    text: str

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]

class QARequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str = Field(..., example="What is recursion?")
    k: int = Field(default=4, ge=1, le=20)

class Citation(BaseModel):
    chunk_id: str
    score: float

class QAResponse(BaseModel):
    answer: str
    citations: list[Citation]
    turn_count: int


app = FastAPI(
    title="RAG Q&A Service",
    description="Retrieval-Augmented Generation Q&A with document ingestion and citations",
    version="1.0.0",
)

@app.get("/")
def root():
    return {
        "service": "RAG Q&A Service",
        "endpoints": ["/ingest", "/search", "/qa"],
        "chunks_stored": len(chunk_store),
        "sessions_active": len(session_store),
    }

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """
    Ingest a document: chunk it, embed each chunk, and store in memory.
    """
    chunks = chunk_text(request.text, chunk_size=300, overlap=50)

    for idx, text_content in enumerate(chunks):
        chunk_id = f"{request.doc_id}#{idx}"
        embedding = await embed_text(text_content)
        chunk_store[chunk_id] = {
            "doc_id": request.doc_id,
            "chunk_index": idx,
            "text": text_content,
            "embedding": embedding,
        }
    return IngestResponse(doc_id=request.doc_id, chunks_added=len(chunks))

@app.get("/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., description="Search query string"),
    k: int = Query(default=3, ge=1, le=20, description="Number of results to return"),
):
    """
    Embed a query and return the top-k most similar chunks via cosine similarity.
    """
    if not chunk_store:
        raise HTTPException(status_code=404, detail="No documents ingested yet.")

    query_embedding = await embed_text(query)
    top_results = cosine_similarity_search(query_embedding, chunk_store, k=k)

    results = [
        SearchResult(
            chunk_id=chunk_id,
            score=round(score, 4),
            text=chunk_store[chunk_id]["text"],
        )
        for chunk_id, score in top_results
    ]

    return SearchResponse(query=query, results=results)

@app.post("/qa", response_model=QAResponse)
async def question_answer(request: QARequest):
    """
    Retrieve top-k relevant chunks, construct a grounded prompt,
    call the LLM with conversation history, and return an answer with citations.
    """
    if not chunk_store:
        raise HTTPException(status_code=404, detail="No documents ingested yet.")

    query_embedding = await embed_text(request.question)
    top_results = cosine_similarity_search(query_embedding, chunk_store, k=request.k)

    retrieved_chunks = [
        {"chunk_id": cid, "score": score, "text": chunk_store[cid]["text"]}
        for cid, score in top_results
    ]

    if request.session_id not in session_store:
        session_store[request.session_id] = []
    history = session_store[request.session_id]

    answer = await generate_grounded_answer(
        question=request.question,
        retrieved_chunks=retrieved_chunks,
        history=history,
    )

    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": answer})

    citations = [
        Citation(chunk_id=c["chunk_id"], score=round(c["score"], 4))
        for c in retrieved_chunks
    ]

    return QAResponse(
        answer=answer,
        citations=citations,
        turn_count=len(history) // 2,
    )