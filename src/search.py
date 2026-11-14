"""The search_docs tool: query -> top-k chunks with ids, page, and version status hidden from the model.

The model sees the document title, date and page in each result header (as a real user would),
but not the internal `status` flag, so it must learn from the dates and wording which rule is current.
"""
import json
import os

# faiss + torch segfault on macOS with multi-threaded OpenMP
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IDX = os.path.join(ROOT, "data/index")


class Searcher:
    def __init__(self):
        self.index = faiss.read_index(os.path.join(_IDX, "faiss.index"))
        with open(os.path.join(_IDX, "chunks.jsonl"), encoding="utf-8") as f:
            self.chunks = [json.loads(l) for l in f]
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # Hybrid search: MiniLM alone scores almost every legal passage as similar (0.7-0.85),
        # so a TF-IDF keyword ranking is fused with it (reciprocal rank fusion, k=60).
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.tf = self.tfidf.fit_transform([c["text"] for c in self.chunks])

    def search(self, query, k=4, pool=30):
        v = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        dense_scores, dense_ids = self.index.search(v, pool)
        kw = (self.tf @ self.tfidf.transform([query]).T).toarray().ravel()
        kw_ids = kw.argsort()[::-1][:pool]
        fused = {}
        for rank, i in enumerate(dense_ids[0]):
            if i >= 0:
                fused[int(i)] = fused.get(int(i), 0) + 1 / (60 + rank)
        for rank, i in enumerate(kw_ids):
            if kw[i] > 0:
                fused[int(i)] = fused.get(int(i), 0) + 1 / (60 + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [dict(self.chunks[i], score=round(fused[i], 4)) for i in top]

    def render(self, hits):
        """Text the model receives as the tool observation."""
        return "\n\n".join(f"[chunk {h['chunk_id']}] {h['text']}" for h in hits)


if __name__ == "__main__":
    import sys
    s = Searcher()
    for h in s.search(" ".join(sys.argv[1:]) or "moratorium period", 4):
        print(h["chunk_id"], h["status"], h["doc"], "p", h["page"], round(h["score"], 3))
        print("   ", h["text"][:200].replace("\n", " "))
