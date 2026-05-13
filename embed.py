"""
build_index.py
Run this ONCE to create constitution.index and documents.json
Usage: python build_index.py
"""

import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────
with open("final.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} raw entries")


# ─────────────────────────────────────────
# 2. CLEAN CONTENT
#    - Replace %N% [actual text] → actual text
#    - Skip omitted / empty articles
# ─────────────────────────────────────────
def clean_content(text: str) -> str:
    # Replace amendment placeholders: %1% [REAL TEXT] → REAL TEXT
    text = re.sub(r'%\d+%\s*\[([^\]]+)\]', r'\1', text)
    # Collapse extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def is_omitted(text: str) -> bool:
    low = text.lower()
    return (
        low.startswith("omitted") or
        "omitted by the constitution" in low[:120] or
        len(text.strip()) < 30          # nearly empty
    )


# ─────────────────────────────────────────
# 3. BUILD DOCUMENT CHUNKS
#    Each chunk = one article or schedule
#    Format:  "Article 14 | Equality before law\n<content>"
# ─────────────────────────────────────────
chunks = []       # text fed to embedder
metadata = []     # parallel list for display/citation

for entry in raw_data:
    doc_type      = entry.get("document_type", "article")
    art_num       = entry.get("article_number") or ""
    sched_num     = entry.get("schedule_number") or ""
    raw_content   = entry.get("content", "")
    amend_notes   = entry.get("amendment_notes") or {}

    content = clean_content(raw_content)

    if is_omitted(content):
        continue                        # ← drop omitted articles
    if not content:
        continue

    # Build a human-readable label
    if doc_type == "schedule":
        label = f"Schedule {sched_num}"
    elif art_num == "Preamble":
        label = "Preamble"
    else:
        label = f"Article {art_num}"

    # Prepend label so the embedding "knows" what article this is
    chunk_text = f"{label}\n{content}"

    # Optionally append amendment notes as extra context
    if amend_notes:
        notes_text = " | ".join(amend_notes.values())
        chunk_text += f"\n[Amendments: {notes_text}]"

    chunks.append(chunk_text)
    metadata.append({
        "label":      label,
        "doc_type":   doc_type,
        "art_num":    art_num,
        "sched_num":  sched_num,
        "start_page": entry.get("start_page"),
    })

print(f"Clean chunks ready: {len(chunks)}  (dropped {len(raw_data) - len(chunks)} omitted/empty)")


# ─────────────────────────────────────────
# 4. EMBED
# ─────────────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding chunks (this may take a minute)...")
embeddings = model.encode(chunks, show_progress_bar=True, batch_size=64)
embeddings = np.array(embeddings, dtype="float32")

print(f"Embedding shape: {embeddings.shape}")


# ─────────────────────────────────────────
# 5. BUILD FAISS INDEX
#    Use IndexFlatIP (inner product) after
#    L2-normalising → equivalent to cosine sim
# ─────────────────────────────────────────
faiss.normalize_L2(embeddings)          # in-place normalise

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)          # cosine similarity
index.add(embeddings)

faiss.write_index(index, "constitution.index")
print(f"Saved constitution.index  ({index.ntotal} vectors)")


# ─────────────────────────────────────────
# 6. SAVE DOCUMENTS + METADATA
# ─────────────────────────────────────────
with open("documents.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

with open("documents_meta.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("Saved documents.json and documents_meta.json")
print("\n✅ Index build complete. Now run rag.py")