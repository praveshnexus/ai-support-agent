
import logging
import os
import re
from pathlib import Path
from uuid import uuid4

import chromadb
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Persistent vector database stored locally.
BACKEND_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BACKEND_DIR / "data" / "chroma"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "gemini-embedding-001"

_client = None
_collection = None


def get_collection():
    """Initialize ChromaDB once and reuse the collection."""
    global _client, _collection

    if _collection is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name="resolveai_knowledge",
            metadata={"hnsw:space": "cosine"},
        )

    return _collection


def get_gemini_client():
    """Create a Gemini client using the backend environment variable."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30_000),
    )


def extract_pdf_text(file_path: str) -> str:
    """Extract text from a PDF file."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)

    return "\n".join(pages).strip()


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def embed_texts(
    texts: list[str],
    task_type: str,
) -> list[list[float]]:
    """Generate Gemini embeddings for documents or search queries."""
    if not texts:
        return []

    client = get_gemini_client()
    embeddings = []

    # Keep requests small to avoid excessively large API calls.
    for start in range(0, len(texts), 32):
        batch = texts[start : start + 32]

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768,
            ),
        )

        if not response.embeddings:
            raise RuntimeError("Gemini returned no embeddings.")

        if len(response.embeddings) != len(batch):
            raise RuntimeError("Embedding count did not match input count.")

        for item in response.embeddings:
            if not item.values:
                raise RuntimeError("Gemini returned an empty embedding.")

            embeddings.append(list(item.values))

    return embeddings


def add_document(
    filename: str,
    text: str,
) -> dict:
    """Chunk, embed, and save a text document in ChromaDB."""
    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError("The document contains no extractable text.")

    chunks = chunk_text(cleaned_text)

    if not chunks:
        raise ValueError("The document could not be split into chunks.")

    embeddings = embed_texts(
        chunks,
        task_type="RETRIEVAL_DOCUMENT",
    )

    collection = get_collection()
    document_id = str(uuid4())

    collection.add(
        ids=[
            f"{document_id}_{index}"
            for index in range(len(chunks))
        ],
        documents=chunks,
        embeddings=[list(map(float, vector)) for vector in embeddings],
        metadatas=[
            {
                "document_id": document_id,
                "filename": Path(filename).name,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ],
    )

    logger.info(
        "Indexed document %s with %d chunks",
        Path(filename).name,
        len(chunks),
    )

    return {
        "document_id": document_id,
        "filename": Path(filename).name,
        "chunks_indexed": len(chunks),
    }


def search_knowledge(
    question: str,
    top_k: int = 4,
) -> list[dict]:
    """Retrieve relevant knowledge chunks for a question."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = embed_texts(
        [question],
        task_type="RETRIEVAL_QUERY",
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    for document, metadata, distance in zip(
        documents[0],
        metadatas[0],
        distances[0],
    ):
        matches.append(
            {
                "text": document,
                "filename": metadata.get("filename", "Unknown"),
                "chunk_index": metadata.get("chunk_index", 0),
                "distance": distance,
            }
        )

    return matches
