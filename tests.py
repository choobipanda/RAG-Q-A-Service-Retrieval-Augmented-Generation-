"""
Test suite for the RAG Q&A Service.

Covers:
- Unit tests: chunking, cosine similarity
- Endpoint tests: ingest + search workflow
- Mock LLM tests: deterministic QA testing
"""

import math
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

# ─────────────────────────────────────────────────────────
# Unit Tests: Chunker
# ─────────────────────────────────────────────────────────

from chunker import chunk_text


class TestChunker:
    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_returns_single_chunk(self):
        text = "This is a short document."
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text.strip()

    def test_chunk_count_is_correct(self):
        # 1000 words, chunk_size=300, overlap=50 → step=250
        # chunks at: 0, 250, 500, 750 → 4 chunks
        text = " ".join(["word"] * 1000)
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        assert len(chunks) == 4

    def test_chunks_have_correct_word_count(self):
        text = " ".join([f"w{i}" for i in range(1000)])
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        for chunk in chunks[:-1]:  # all but last should be full-sized
            assert len(chunk.split()) == 300

    def test_overlap_is_present(self):
        words = [f"w{i}" for i in range(400)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        # Last 50 words of chunk 0 == first 50 words of chunk 1
        tail_of_first = chunks[0].split()[-50:]
        head_of_second = chunks[1].split()[:50]
        assert tail_of_first == head_of_second

    def test_no_content_lost(self):
        words = [f"w{i}" for i in range(500)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        # Every word from the original must appear in at least one chunk
        all_chunk_words = set()
        for chunk in chunks:
            all_chunk_words.update(chunk.split())
        assert all_chunk_words == set(words)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            chunk_text("hello world", chunk_size=100, overlap=100)


# ─────────────────────────────────────────────────────────
# Unit Tests: Cosine Similarity
# ─────────────────────────────────────────────────────────

from retrieval import cosine_similarity, cosine_similarity_search


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_return_minus_one(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_known_similarity(self):
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1 / math.sqrt(2)
        assert cosine_similarity(a, b) == pytest.approx(expected, rel=1e-5)

    def test_search_returns_top_k(self):
        store = {
            "doc#0": {"text": "a", "embedding": [1.0, 0.0]},
            "doc#1": {"text": "b", "embedding": [0.9, 0.1]},
            "doc#2": {"text": "c", "embedding": [0.0, 1.0]},
            "doc#3": {"text": "d", "embedding": [-1.0, 0.0]},
        }
        query = [1.0, 0.0]
        results = cosine_similarity_search(query, store, k=2)
        assert len(results) == 2
        assert results[0][0] == "doc#0"  # most similar
        assert results[1][0] == "doc#1"  # second

    def test_search_results_sorted_descending(self):
        store = {
            f"doc#{i}": {"text": "x", "embedding": [float(i), 0.0]}
            for i in range(1, 6)
        }
        query = [5.0, 0.0]
        results = cosine_similarity_search(query, store, k=5)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


# ─────────────────────────────────────────────────────────
# Endpoint Tests (with mocked OpenAI)
# ─────────────────────────────────────────────────────────

SAMPLE_DOC = {
    "doc_id": "cs_notes",
    "text": (
        "Recursion is a programming technique where a function calls itself. "
        "Base case stops the recursion. Recursive algorithms are used in tree traversal. "
        "Dynamic programming can replace recursion for efficiency. "
        "Stack overflow can occur with deep recursion. "
        "Tail recursion is an optimization supported by some compilers. "
        "Fibonacci sequence is a classic recursion example. "
    ) * 20,  # repeat to create multiple chunks
}

MOCK_EMBEDDING = [0.1] * 1536  # text-embedding-3-small output size
MOCK_ANSWER = "Recursion is when a function calls itself, as described in [cs_notes#0]."


def make_test_client():
    """Create a fresh TestClient with cleared stores."""
    from store import chunk_store, session_store
    chunk_store.clear()
    session_store.clear()
    from main import app
    return TestClient(app)


@pytest.fixture
def client_with_mocks():
    """TestClient with OpenAI calls mocked out."""
    with (
        patch("embeddings.AsyncOpenAI") as mock_embed_client_cls,
        patch("llm.AsyncOpenAI") as mock_llm_client_cls,
    ):
        # Mock embed_text
        mock_embed_instance = AsyncMock()
        mock_embed_instance.embeddings.create.return_value = AsyncMock(
            data=[AsyncMock(embedding=MOCK_EMBEDDING)]
        )
        mock_embed_client_cls.return_value = mock_embed_instance

        # Mock chat completions
        mock_llm_instance = AsyncMock()
        mock_llm_instance.chat.completions.create.return_value = AsyncMock(
            choices=[AsyncMock(message=AsyncMock(content=MOCK_ANSWER))]
        )
        mock_llm_client_cls.return_value = mock_llm_instance

        # Reset module-level cached clients
        import embeddings
        import llm
        embeddings._client = None
        llm._client = None

        yield make_test_client()


class TestIngestEndpoint:
    def test_ingest_returns_correct_doc_id(self, client_with_mocks):
        resp = client_with_mocks.post("/ingest", json=SAMPLE_DOC)
        assert resp.status_code == 200
        assert resp.json()["doc_id"] == "cs_notes"

    def test_ingest_returns_positive_chunk_count(self, client_with_mocks):
        resp = client_with_mocks.post("/ingest", json=SAMPLE_DOC)
        assert resp.json()["chunks_added"] > 0

    def test_ingest_chunk_count_matches_expected(self, client_with_mocks):
        from chunker import chunk_text
        expected = len(chunk_text(SAMPLE_DOC["text"]))
        resp = client_with_mocks.post("/ingest", json=SAMPLE_DOC)
        assert resp.json()["chunks_added"] == expected

    def test_ingest_short_doc_returns_one_chunk(self, client_with_mocks):
        resp = client_with_mocks.post("/ingest", json={
            "doc_id": "tiny",
            "text": "Short document.",
        })
        assert resp.json()["chunks_added"] == 1


class TestSearchEndpoint:
    def test_search_without_docs_returns_404(self, client_with_mocks):
        resp = client_with_mocks.get("/search?query=recursion")
        assert resp.status_code == 404

    def test_search_after_ingest_returns_results(self, client_with_mocks):
        client_with_mocks.post("/ingest", json=SAMPLE_DOC)
        resp = client_with_mocks.get("/search?query=recursion&k=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "recursion"
        assert len(data["results"]) == 3

    def test_search_result_fields(self, client_with_mocks):
        client_with_mocks.post("/ingest", json=SAMPLE_DOC)
        resp = client_with_mocks.get("/search?query=test&k=2")
        for result in resp.json()["results"]:
            assert "chunk_id" in result
            assert "score" in result
            assert "text" in result

    def test_search_k_limits_results(self, client_with_mocks):
        client_with_mocks.post("/ingest", json=SAMPLE_DOC)
        for k in [1, 2, 5]:
            resp = client_with_mocks.get(f"/search?query=test&k={k}")
            assert len(resp.json()["results"]) == k


class TestQAEndpoint:
    def test_qa_without_docs_returns_404(self, client_with_mocks):
        resp = client_with_mocks.post("/qa", json={
            "session_id": "test-session",
            "question": "What is recursion?",
        })
        assert resp.status_code == 404

    def test_qa_returns_answer_and_citations(self, client_with_mocks):
        client_with_mocks.post("/ingest", json=SAMPLE_DOC)
        resp = client_with_mocks.post("/qa", json={
            "session_id": "sess-1",
            "question": "What is recursion?",
            "k": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "citations" in data
        assert data["answer"] == MOCK_ANSWER

    def test_qa_citation_fields(self, client_with_mocks):
        client_with_mocks.post("/ingest", json=SAMPLE_DOC)
        resp = client_with_mocks.post("/qa", json={
            "session_id": "sess-2",
            "question": "Explain base case.",
            "k": 2,
        })
        for cite in resp.json()["citations"]:
            assert "chunk_id" in cite
            assert "score" in cite

    def test_qa_turn_count_increments(self, client_with_mocks):
        client_with_mocks.post("/ingest", json=SAMPLE_DOC)
        session_id = "sess-turn-test"
        for expected_turn in range(1, 4):
            resp = client_with_mocks.post("/qa", json={
                "session_id": session_id,
                "question": "Tell me about recursion.",
                "k": 2,
            })
            assert resp.json()["turn_count"] == expected_turn

    def test_qa_creates_session_automatically(self, client_with_mocks):
        client_with_mocks.post("/ingest", json=SAMPLE_DOC)
        # No session_id provided — should auto-generate one
        resp = client_with_mocks.post("/qa", json={
            "question": "What is a base case?",
        })
        assert resp.status_code == 200
        assert resp.json()["turn_count"] == 1
