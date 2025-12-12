import os
from typing import Iterable, Tuple

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


INDEX_NAME = "log-anomalies-index"


def get_pinecone() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY") or "pclocal"
    return Pinecone(api_key=api_key)


def ensure_index(pc: Pinecone, dimension: int, metric: str = "cosine"):
    existing = {idx.name for idx in pc.list_indexes()}
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def upsert_vectors(pc: Pinecone, items: Iterable[Tuple[str, list, dict]]):
    index = pc.Index(INDEX_NAME)
    index.upsert(vectors=[{"id": vid, "values": vec, "metadata": meta} for vid, vec, meta in items])
