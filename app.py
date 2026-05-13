import re
import json
import numpy as np
import os
from dotenv import load_dotenv
import faiss
from flask import Flask, request, jsonify, render_template
from groq import Groq
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
load_dotenv()
# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_FILE       = "final.json"
INDEX_FILE      = "constitution.index"
EMBED_MODEL     = "all-MiniLM-L6-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"
FAISS_TOP_K     = 6
FAISS_THRESHOLD = 0.25

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embed_model = SentenceTransformer(EMBED_MODEL)

# ─────────────────────────────────────────
# LOAD DATA FROM final.json
# ─────────────────────────────────────────
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

documents = []
metadata  = []

for entry in raw_data:
    content = entry.get("content", "").strip()
    if not content:
        continue

    art_num  = entry.get("article_number")
    sched_no = entry.get("schedule_number")

    if art_num == "Preamble":
        label = "Preamble"
    elif art_num is not None:
        label = f"Article {art_num}"
    elif sched_no is not None:
        label = f"Schedule {sched_no}"
    else:
        label = "Unknown"

    documents.append(content)
    metadata.append({
        "article_number":  art_num,
        "schedule_number": sched_no,
        "label":           label,
        "amendment_notes": entry.get("amendment_notes", {}),
        "start_page":      entry.get("start_page"),
    })

print(f"[INIT] Loaded {len(documents)} chunks from {DATA_FILE}")

# ─────────────────────────────────────────
# BUILD LOOKUP TABLES
# ─────────────────────────────────────────
article_lookup  = {}
schedule_lookup = {}

for i, meta in enumerate(metadata):
    art = meta["article_number"]
    sch = meta["schedule_number"]
    if art is not None:
        key = str(art).strip()
        if key not in article_lookup:
            article_lookup[key] = i
    if sch is not None:
        schedule_lookup.setdefault(sch, []).append(i)

print(f"[INIT] Articles indexed: {len(article_lookup)}  (Preamble: {'Preamble' in article_lookup})")
print(f"[INIT] Schedules indexed: {len(schedule_lookup)}")

# ─────────────────────────────────────────
# LOAD / BUILD FAISS INDEX
# ─────────────────────────────────────────
try:
    faiss_index = faiss.read_index(INDEX_FILE)
    if faiss_index.ntotal != len(documents):
        raise ValueError(f"Index has {faiss_index.ntotal} vectors but {len(documents)} chunks — rebuilding")
    print(f"[INIT] Loaded FAISS index ({faiss_index.ntotal} vectors)")
except Exception as e:
    print(f"[INIT] Building FAISS index: {e}")
    embeddings = embed_model.encode(documents, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(embeddings)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, INDEX_FILE)
    print(f"[INIT] FAISS index built & saved ({faiss_index.ntotal} vectors)")

# ─────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────
SYSTEM_PROMPT = """You are SamvidhanGPT, an expert AI assistant dedicated exclusively to the Indian Constitution.

RULES (never break these):
1. Answer ONLY from the CONSTITUTIONAL CONTEXT block provided. Never use outside knowledge.
2. If the answer is not in the context, respond with exactly:
   "I'm sorry, that information isn't in the retrieved context. Please refer to https://legislative.gov.in for the complete Constitution."
3. Always cite the exact Article, Schedule, or Part your answer comes from.
4. If amendment notes are present in the context, mention the relevant ones.
5. Match the user's requested format — if they ask for bullet points or a numbered list, use that format.
6. Be concise. No hallucination. No general knowledge. No disclaimers like "generally speaking"."""

OFF_TOPIC_MSG = (
    "🙏 I am **SamvidhanGPT**, designed exclusively to answer questions about the "
    "**Indian Constitution** — its Articles, Schedules, Amendments, Fundamental Rights, "
    "Directive Principles, and constitutional officeholders.\n\n"
    "Your question appears to be outside this scope. Please ask something "
    "related to the Indian Constitution and I'll be happy to help!"
)

# ─────────────────────────────────────────
# TOPIC GUARD — keyword bypass + LLM fallback
# ─────────────────────────────────────────
BYPASS_WORDS = [
    "article", "schedule", "amendment", "preamble", "fundamental",
    "directive", "constitution", "parliament", "president", "judiciary",
    "citizenship", "election", "governor", "prime minister", "chief justice",
    "speaker", "attorney general", "lok sabha", "rajya sabha",
    "supreme court", "high court", "union", "state list", "concurrent list",
    "fundamental rights", "fundamental duties", "right to", "part iii",
    "part iv", "part v", "part vi", "writ", "habeas corpus", "mandamus",
    "certiorari", "prohibition", "quo warranto", "dpsp", "federal",
]

GUARD_PROMPT = """You are a strict topic classifier for a chatbot that ONLY answers questions about the Indian Constitution.

Classify the user's question as YES (related to Indian Constitution, its articles, schedules, amendments, government structure, constitutional officeholders, fundamental rights, directive principles, elections, or citizenship) or NO (anything else).

Reply with exactly one word: YES or NO."""

def is_constitutional(query: str) -> bool:
    q = query.lower()
    if any(w in q for w in BYPASS_WORDS):
        print(f"[TOPIC GUARD] '{query[:60]}' → BYPASS (keyword match)")
        return True
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": GUARD_PROMPT},
                {"role": "user",   "content": query},
            ],
            temperature=0.0, max_tokens=5,
        )
        verdict = r.choices[0].message.content.strip().upper()
        print(f"[TOPIC GUARD] '{query[:60]}' → {verdict}")
        return verdict.startswith("YES")
    except Exception as e:
        print(f"[TOPIC GUARD ERROR] {e} — failing open")
        return True

# ─────────────────────────────────────────
# QUERY PARSERS
# ─────────────────────────────────────────
PREAMBLE_KEYWORDS = [
    "preamble", "we the people", "sovereign", "socialist", "secular",
    "democratic republic", "liberty of thought", "equality of status",
    "fraternity", "constituent assembly", "justice social",
]

SCHEDULE_WORDS = {
    "first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,
    "seventh":7,"eighth":8,"ninth":9,"tenth":10,"eleventh":11,"twelfth":12,
    "1st":1,"2nd":2,"3rd":3,"4th":4,"5th":5,"6th":6,
    "7th":7,"8th":8,"9th":9,"10th":10,"11th":11,"12th":12,
}

def is_preamble_query(q: str) -> bool:
    return any(kw in q.lower() for kw in PREAMBLE_KEYWORDS)

def extract_article_numbers(q: str) -> list:
    matches = re.findall(r'\b(?:article|art\.?)\s*(\d+[A-Za-z]*)\b', q, re.IGNORECASE)
    return [m[:-1] + m[-1].upper() if m[-1].isalpha() else m for m in matches]

def extract_schedule_numbers(q: str) -> list:
    found = []
    ql = q.lower()
    for m in re.finditer(r'\bschedule\s+(\d+)\b|\b(\d+)(?:st|nd|rd|th)?\s+schedule\b', ql):
        n = int(m.group(1) or m.group(2))
        if 1 <= n <= 12:
            found.append(n)
    for word, num in SCHEDULE_WORDS.items():
        if re.search(rf'\b{word}\s+schedule\b|\bschedule\s+{word}\b', ql):
            found.append(num)
    return list(set(found))

# ─────────────────────────────────────────
# CONTEXT BUILDERS
# ─────────────────────────────────────────
def fmt(meta: dict, content: str, source: str = "direct") -> str:
    notes = meta.get("amendment_notes", {})
    text  = f"[{meta['label']}] ({source})\n{content}"
    if notes:
        text += "\n\nAmendment Notes:\n" + "\n".join(
            f"  [{k}]: {v}" for k, v in notes.items()
        )
    return text + "\n\n---\n\n"

def ctx_preamble() -> str:
    idx = article_lookup.get("Preamble") or article_lookup.get("preamble")
    if idx is not None:
        print(f"[LOOKUP] Preamble → index {idx}")
        return fmt(metadata[idx], documents[idx])
    print("[LOOKUP] Preamble NOT FOUND in final.json!")
    return ""

def ctx_articles(art_nums: list) -> str:
    out = ""
    for n in art_nums:
        idx = article_lookup.get(n) or article_lookup.get(n.upper())
        if idx is not None:
            print(f"[LOOKUP] Article {n} → index {idx}")
            out += fmt(metadata[idx], documents[idx])
        else:
            print(f"[LOOKUP] Article {n} NOT FOUND")
    return out

def ctx_schedules(sched_nums: list) -> str:
    out = ""
    for n in sched_nums:
        idxs = schedule_lookup.get(n, [])
        if idxs:
            print(f"[LOOKUP] Schedule {n} → {len(idxs)} chunk(s)")
            for idx in idxs:
                out += fmt(metadata[idx], documents[idx])
        else:
            print(f"[LOOKUP] Schedule {n} NOT FOUND")
    return out

def ctx_faiss(query: str, already: set) -> str:
    qv = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(qv)
    dists, idxs = faiss_index.search(qv, FAISS_TOP_K)
    out = ""
    for rank, idx in enumerate(idxs[0]):
        if idx == -1:
            continue
        score = float(dists[0][rank])
        if score < FAISS_THRESHOLD:
            print(f"[FAISS] Low score — skipping idx={idx} score={score:.3f}")
            continue
        meta  = metadata[idx]
        a_key = str(meta["article_number"])  if meta["article_number"]  is not None else None
        s_key = str(meta["schedule_number"]) if meta["schedule_number"] is not None else None
        if a_key in already or s_key in already:
            continue
        out += fmt(meta, documents[idx], source=f"semantic {score:.3f}")
        if a_key: already.add(a_key)
        if s_key: already.add(s_key)
    return out

# ─────────────────────────────────────────
# OFFICEHOLDER DETECTION & LOOKUP
# ─────────────────────────────────────────
OFFICEHOLDER_KEYWORDS = [
    "who is", "who was", "current", "present", "former", "ex-",
    "president of india", "prime minister", "chief minister",
    "chief justice", "governor", "vice president",
    "speaker", "attorney general", "solicitor general",
    "home minister", "finance minister", "defence minister",
    "cabinet minister", "external affairs", "name of",
]

def needs_officeholder_info(query: str) -> bool:
    return any(kw in query.lower() for kw in OFFICEHOLDER_KEYWORDS)

def get_officeholder_info(query: str) -> str:
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a factual assistant with knowledge about Indian government officials. "
                        "Give the person's full name, exact title, and when they took office. "
                        "Be direct and factual. No disclaimers."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.0, max_tokens=200,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[OFFICEHOLDER ERROR] {e}")
        return ""

# ─────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get-response', methods=['POST'])
def get_response():
    data       = request.json
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"response": "Please ask a question."})

    print(f"\n[USER] {user_input}")

    # ── Step 0: Topic guard ───────────────────────────────────────
    if not is_constitutional(user_input):
        print("[BLOCKED] Off-topic query rejected.")
        return jsonify({"response": OFF_TOPIC_MSG})

    # ── Step 1: Direct lookups ────────────────────────────────────
    art_nums   = extract_article_numbers(user_input)
    sched_nums = extract_schedule_numbers(user_input)
    already    = set(art_nums)
    already.update(str(n) for n in sched_nums)

    direct_ctx = ""

    if is_preamble_query(user_input):
        print("[INFO] Preamble query detected")
        direct_ctx += ctx_preamble()
        already.add("Preamble")

    if art_nums:
        print(f"[INFO] Articles detected: {art_nums}")
        direct_ctx += ctx_articles(art_nums)

    if sched_nums:
        print(f"[INFO] Schedules detected: {sched_nums}")
        direct_ctx += ctx_schedules(sched_nums)

    # ── Step 2: FAISS semantic search ─────────────────────────────
    faiss_ctx = ctx_faiss(user_input, already)

    # ── Step 3: Combine context ───────────────────────────────────
    constitution_ctx = (direct_ctx + faiss_ctx).strip()

    # ── Step 4: Officeholder info if needed ───────────────────────
    officeholder_ctx = ""
    if needs_officeholder_info(user_input):
        print("[INFO] Fetching officeholder info...")
        officeholder_ctx = get_officeholder_info(user_input)
        print(f"[OFFICEHOLDER] {officeholder_ctx[:150]}")

    # ── Step 5: Build prompt ──────────────────────────────────────
    user_msg = f"--- CONSTITUTIONAL CONTEXT ---\n{constitution_ctx}\n--- END CONTEXT ---\n"
    if officeholder_ctx:
        user_msg += f"\n--- CURRENT OFFICEHOLDER INFO ---\n{officeholder_ctx}\n--- END OFFICEHOLDER INFO ---\n"
    user_msg += f"\nQuestion: {user_input}"

    # ── Step 6: LLM answer ────────────────────────────────────────
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    print(f"[AI] {answer[:200]}...")
    return jsonify({"response": answer})


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)