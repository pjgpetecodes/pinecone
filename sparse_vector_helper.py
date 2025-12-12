import re
from typing import Dict, List

from rank_bm25 import BM25Okapi


class SparseVectorHelper:
    """Generate sparse vectors using simple BM25 over a dynamically built corpus."""

    def __init__(self, vocab_mod: int = 30000):
        self.vocab_mod = vocab_mod  # keep token ids bounded for demo
        self.corpus_tokens: List[List[str]] = []
        self.bm25 = None

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if t]

    def add_to_corpus(self, documents: List[str]):
        tokenized = [self.tokenize(doc) for doc in documents]
        self.corpus_tokens.extend(tokenized)
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def get_sparse_vector(self, text: str) -> Dict[int, float]:
        if not self.bm25:
            raise RuntimeError("Sparse corpus not initialized. Call add_to_corpus first.")
        tokens = self.tokenize(text)
        # BM25 scores per doc; aggregate token presence into sparse vector
        token_scores: Dict[str, float] = {}
        for tok in set(tokens):
            score = 0.0
            for doc_tokens in self.corpus_tokens:
                if tok in doc_tokens:
                    score += 1.0  # simple presence weighting; demo-friendly
            if score > 0:
                token_scores[tok] = score

        sparse = {}
        for tok, score in token_scores.items():
            tok_id = hash(tok) % self.vocab_mod
            # Cap scores to 1.0 for demo stability
            sparse[tok_id] = min(score, 1.0)
        return sparse

    def text_to_sparse_tf(self, text: str) -> Dict[int, float]:
        tokens = self.tokenize(text)
        counts: Dict[int, int] = {}
        for tok in tokens:
            tok_id = hash(tok) % self.vocab_mod
            counts[tok_id] = counts.get(tok_id, 0) + 1
        max_c = max(counts.values()) if counts else 1
        return {tid: c / max_c for tid, c in counts.items()}
