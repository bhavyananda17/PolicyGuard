"""Chunk the IRDAI documents and build a FAISS index with a version tag on every chunk.

Same chunking choices as our earlier study index (about 140 words per chunk, 25-word
overlap, header line with the document title, MiniLM embeddings, cosine search), adapted
for PDF text: text is cleaned page by page and every chunk records its page number.

    python src/build_index.py            # reads data/text/, writes data/index/
"""
import json
import os

# faiss + torch segfault on macOS with multi-threaded OpenMP
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED_MODEL = "all-MiniLM-L6-v2"
TARGET_WORDS = 140
OVERLAP_WORDS = 25
MAX_TOKENS = 256

# status: "current" = in force today; "superseded" = replaced, kept as a realistic trap
DOCS = {
    "current_mc_health_2024": {
        "title": "IRDAI Master Circular on Health Insurance Business (29 May 2024)",
        "ref": "IRDAI/HLT/CIR/PRO/84/5/2024", "date": "2024-05-29", "status": "current",
        "source": "https://irdai.gov.in/document-detail?documentId=4942918",
    },
    "current_mc_health_2024_annexure": {
        "title": "Annexure to IRDAI Master Circular on Health Insurance Business (29 May 2024)",
        "ref": "IRDAI/HLT/CIR/PRO/84/5/2024", "date": "2024-05-29", "status": "current",
        "source": "https://irdai.gov.in/document-detail?documentId=4942918",
    },
    "superseded_mc_standardization_2020": {
        "title": "IRDAI Master Circular on Standardization of Health Insurance Products (22 July 2020)",
        "ref": "IRDAI/HLT/REG/CIR/193/07/2020", "date": "2020-07-22", "status": "superseded",
        "superseded_by": "IRDAI/HLT/CIR/PRO/84/5/2024",
        "source": "https://irdai.gov.in/document-detail?documentId=395035",
    },
}

PAGE_MARK = re.compile(r"^\s*Page \d+ of \d+\s*$", re.M)
SENT = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z(\"'0-9])")


def clean(page):
    page = PAGE_MARK.sub("", page)
    page = re.sub(r"[ \t]+", " ", page)
    page = re.sub(r"-\n(?=[a-z])", "", page)       # re-join hyphenated line breaks
    page = re.sub(r"\s*\n\s*", " ", page)          # PDF line wraps are not paragraphs
    return page.strip()


def sentences(text):
    out = []
    for s in SENT.split(text):
        s = s.strip()
        while len(s.split()) > TARGET_WORDS:       # very long run-on (tables): hard split
            w = s.split()
            out.append(" ".join(w[:TARGET_WORDS]))
            s = " ".join(w[TARGET_WORDS:])
        if s:
            out.append(s)
    return out


def chunk_doc(pages):
    units = [(pno, s) for pno, txt in pages for s in sentences(txt)]
    chunks, cur = [], []
    for pno, s in units:
        if cur and sum(len(x.split()) for _, x in cur) + len(s.split()) > TARGET_WORDS:
            chunks.append(cur)
            lead, n = [], 0                         # overlap: last whole sentences
            for u in reversed(cur):
                lead.insert(0, u)
                n += len(u[1].split())
                if n >= OVERLAP_WORDS:
                    break
            cur = lead if n <= 2 * OVERLAP_WORDS else []
        cur.append((pno, s))
    if cur:
        chunks.append(cur)
    return chunks


def main():
    out_dir = os.path.join(ROOT, "data/index")
    os.makedirs(out_dir, exist_ok=True)
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    tok = model.tokenizer

    import pypdf
    records = []
    for key, meta in DOCS.items():
        reader = pypdf.PdfReader(os.path.join(ROOT, "data/raw", key + ".pdf"))
        pages = [(i + 1, clean(p.extract_text() or "")) for i, p in enumerate(reader.pages)]
        for i, ch in enumerate(chunk_doc(pages)):
            body = " ".join(s for _, s in ch)
            text = f"{meta['title']} | page {ch[0][0]}\n{body}"
            ids = tok(text)["input_ids"]           # over MAX_TOKENS is counted in meta.json
            records.append({
                "chunk_id": len(records), "doc": key, "chunk_in_doc": i,
                "page": ch[0][0], "status": meta["status"], "ref": meta["ref"], "date": meta["date"],
                "text": text, "words": len(body.split()), "tokens": len(ids),
            })

    vecs = model.encode([r["text"] for r in records], batch_size=64, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = {
        "documents": DOCS, "chunks": len(records),
        "current_chunks": sum(r["status"] == "current" for r in records),
        "superseded_chunks": sum(r["status"] == "superseded" for r in records),
        "embedding_model": EMBED_MODEL, "target_words": TARGET_WORDS, "overlap_words": OVERLAP_WORDS,
        "chunks_over_max_tokens": sum(r["tokens"] > MAX_TOKENS for r in records),
        "median_words": int(np.median([r["words"] for r in records])),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(json.dumps({k: v for k, v in info.items() if k != "documents"}, indent=2))


if __name__ == "__main__":
    main()
