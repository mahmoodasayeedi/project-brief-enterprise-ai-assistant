#import 
import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import re
SKU_RE = r"\b[A-Z]{2,4}(?:-[A-Z0-9]{1,6}){2,4}\b"

#1. LOAD — read many file types into plain text
TEXT_EXT = {".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml", ".html", ".htm", ".xml", ".log"}


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
        if ext == ".docx":
            import docx2txt
            return docx2txt.process(str(path))
        if ext in {".html", ".htm"}:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            return re.sub(r"<[^>]+>", " ", raw)  # strip tags
        if ext in TEXT_EXT or ext == "":
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:  # never crash on one bad file
        print(f"  ⚠️  Could not read {path.name}: {e}")
        return ""
    print(f"  ⏭️  Skipping unsupported file type: {path.name}")
    return ""

# 2. METADATA — department, date, version, title
DEPARTMENTS = {
    "HR": ["hr", "human resources", "leave", "vacation", "payroll", "onboarding", "benefits", "employee handbook"],
    "IT": ["it", "password", "vpn", "laptop", "software", "security", "helpdesk", "access"],
    "Finance": ["finance", "expense", "invoice", "budget", "reimbursement", "procurement", "accounting"],
    "Sales": ["sales", "pricing", "discount", "customer", "crm", "quota", "deal"],
    "Legal": ["legal", "compliance", "contract", "gdpr", "privacy", "policy violation"],
    "Operations": ["operations", "warehouse", "shipping", "logistics", "inventory", "store", "supply"],
    "Marketing": ["marketing", "brand", "campaign", "social media"],
    "Customer Support": ["support", "returns", "refund", "ticket", "complaint"],
    "Engineering": ["engineering", "deployment", "api", "code review", "incident"],
}

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})


def _safe_date(y, m, d):
    try:
        return date(int(y), int(m), int(d))
    except ValueError:
        return None


def find_dates(text: str):
    """Return all dates found in text (ISO, '15 March 2024', 'March 15, 2024', 'Q3 2023', '2024')."""
    found = []
    for y, m, d in re.findall(r"\b(20\d{2}|19\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b", text):
        found.append(_safe_date(y, m, d))
    for d, m, y in re.findall(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](20\d{2})\b", text):  # 15/03/2024
        found.append(_safe_date(y, m, d) or _safe_date(y, d, m))
    month_re = "|".join(sorted(MONTHS, key=len, reverse=True))
    for d, m, y in re.findall(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_re})\.?,?\s+(20\d{{2}})\b", text, re.I):
        found.append(_safe_date(y, MONTHS[m.lower()], d))
    for m, d, y in re.findall(rf"\b({month_re})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(20\d{{2}})\b", text, re.I):
        found.append(_safe_date(y, MONTHS[m.lower()], d))
    for m, y in re.findall(rf"\b({month_re})\.?\s+(20\d{{2}})\b", text, re.I):
        found.append(_safe_date(y, MONTHS[m.lower()], 1))
    return [f for f in found if f]


def extract_date(name: str, text: str):
    """Best guess of when a document was written/updated."""
    # Priority 1: an explicit 'Last updated / Effective / Date' line
    m = re.search(r"(last\s+updated|updated|effective(?:\s+date)?|revised|revision date|date|published)\s*[:\-]\s*(.{0,40})",
                  text[:3000], re.I)
    if m:
        ds = find_dates(m.group(2))
        if ds:
            return max(ds)
    # Priority 2: date in the file name
    ds = find_dates(name.replace("_", " "))
    if ds:
        return max(ds)
    years = re.findall(r"(?<!\d)(20\d{2})(?!\d)", name)
    if years:
        return date(int(max(years)), 1, 1)
    # Priority 3: latest date near the top of the document
    ds = [d for d in find_dates(text[:2000]) if d <= date.today()]
    return max(ds) if ds else None


def extract_version(name: str, text: str):
    m = re.search(r"(?<![a-z])v(?:ersion)?\.?\s*(\d+(?:\.\d+)*)(?![\d.])", name + " " + text[:2000], re.I)
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


def extract_department(path: Path, text: str) -> str:
    docs_root = Path(config.DOCS_DIR).resolve()
    # Folder name wins, e.g. documents/HR/leave_policy.md
    for part in path.resolve().relative_to(docs_root).parts[:-1]:
        for dept, words in DEPARTMENTS.items():
            if part.lower() in [dept.lower()] + words:
                return dept
    m = re.search(r"department\s*[:\-]\s*([A-Za-z &]+)", text[:1500], re.I)
    if m:
        val = m.group(1).strip().lower()
        for dept, words in DEPARTMENTS.items():
            if val.startswith(dept.lower()) or any(val.startswith(w) for w in words if len(w) > 2):
                return dept
    # Otherwise: count keywords in file name (x3) + text
    hay = (path.stem.replace("_", " ").lower() + " ") * 3 + text[:4000].lower()
    scores = {d: sum(len(re.findall(rf"\b{re.escape(w)}\b", hay)) for w in ws) for d, ws in DEPARTMENTS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"


def extract_title(path: Path, text: str) -> str:
    for line in text.splitlines()[:15]:
        line = line.strip().lstrip("#").strip()
        if 3 < len(line) < 120:
            return line
    return path.stem.replace("_", " ").title()


# 3. CLEAN — exact duplicates, near duplicates, outdated versions
def normalize(text: str) -> str:
    text = text.replace("\x7f", "•")
    return re.sub(r"\s+", " ", text.lower()).strip()


def shingles(text: str, n: int = 5) -> set:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


NOISE_WORDS = r"\b(v\d+(\.\d+)*|version|final|draft|old|new|latest|updated|update|copy|rev\d*|revised|archive|archived|backup|current|deprecated)\b"

def family_key(stem: str) -> str:
    """'Leave_Policy_v2_2024-final' -> 'leave policy'  (so versions of one doc group together)."""
    s = stem.lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"\b(19|20)\d{2}\b|\b\d{1,2}\b|\(\d+\)", " ", s)
    s = re.sub(NOISE_WORDS, " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


OUTDATED_HINTS = re.compile(r"\b(deprecated|obsolete|superseded|no longer (valid|in effect|applies)|archived|outdated|replaced by)\b", re.I)


def analyse(docs: list[dict]) -> dict:
    """Mark duplicates + outdated docs. Returns a report dict."""
    report = {"exact_duplicates": [], "families": [], "near_duplicates": [], "flagged_outdated_text": [],
              "old_by_date": [], "sku_conflicts": {}}

    # (a) exact duplicates -> keep the first, drop the rest
    seen = {}
    # keep the "cleanest" file name (not 'copy', 'backup'...) when duplicates exist
    for d in sorted(docs, key=lambda x: (bool(re.search(NOISE_WORDS + r"|\(\d+\)", x["source"].lower())), len(x["source"]))):
        h = hashlib.md5(normalize(d["text"]).encode()).hexdigest()
        if h in seen:
            d["status"] = "duplicate"
            d["duplicate_of"] = seen[h]["source"]
            report["exact_duplicates"].append((d["source"], seen[h]["source"]))
        else:
            seen[h] = d
    live = [d for d in docs if d["status"] != "duplicate"]

    # (b) group versions of the same document (same 'family' name OR >60% same content)
    parent = list(range(len(live)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for d in live:
        d["_sh"] = shingles(d["text"])
    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            a, b = live[i], live[j]
            same_name = a["family"] and a["family"] == b["family"]
            sim = jaccard(a["_sh"], b["_sh"])
            if same_name or sim > 0.6:
                parent[find(i)] = find(j)
                if sim > 0.6:
                    report["near_duplicates"].append((a["source"], b["source"], round(sim, 2)))

    groups = {}
    for i, d in enumerate(live):
        groups.setdefault(find(i), []).append(d)

    # (c) inside each group, newest = current, others = outdated
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda d: (d["date"] or "0000", d["version"], d["mtime"]), reverse=True)
        newest = members[0]
        for old in members[1:]:
            old["status"] = "outdated"
            old["superseded_by"] = newest["source"]
        report["families"].append([(m["source"], m["date"], ".".join(map(str, m["version"])) or "-", m["status"]) for m in members])

    # (d) documents that SAY they are outdated
    for d in live:
        if d["status"] == "current" and OUTDATED_HINTS.search(d["text"][:1500]):
            report["flagged_outdated_text"].append(d["source"])
            d["warning"] = "Document text says it may be deprecated/superseded."
    # (e) documents much older than the rest of the collection (e.g. Old_Product_Catalog_2021)
    dates = [d["date"] for d in live if d["date"]]
    if dates:
        newest = max(dates)
        for d in live:
            if d["status"] == "current" and d["date"] and int(newest[:4]) - int(d["date"][:4]) >= 2:
                d["status"] = "outdated"
                d["warning"] = f"Last reviewed {d['date']}, much older than the rest of the documents ({newest})."
                report["old_by_date"].append(d["source"])

    # (f) the same SKU used for different products in different files
    sku_files = {}
    for d in live:
        for sku in set(re.findall(r"\bIS-[A-Z]+-[A-Z0-9]+(?:-[A-Z0-9]+)*\b", d["text"])):
            sku_files.setdefault(sku, []).append(d["source"])
    report["sku_conflicts"] = {s: f for s, f in sku_files.items() if len(f) > 1}

    for d in live:
        d.pop("_sh", None)
    return report

# 4. CHUNK
def make_chunks(docs: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks, seen = [], set()
    for d in docs:
        if d["status"] == "duplicate" or not d["text"].strip():
            continue
        for i, piece in enumerate(splitter.split_text(d["text"])):
            h = hashlib.md5(normalize(piece).encode()).hexdigest()
            if h in seen:  # same paragraph copied into several docs -> keep once
                continue
            seen.add(h)
            header = f"[{d['title']} | Dept: {d['department']} | Date: {d['date'] or 'unknown'}]\n"
            chunks.append(Document(
                page_content=header + piece,
                metadata={
                    "source": d["source"], "title": d["title"], "department": d["department"],
                    "date": d["date"] or "unknown", "status": d["status"],
                    "superseded_by": d.get("superseded_by", ""), "warning": d.get("warning", ""),
                    "chunk": i,
                },
            ))
    return chunks


# REPORT — great to show your teacher what the data looked like
def write_report(docs, report, n_chunks):
    lines = ["# IronStore Documents — Data Quality Report\n",
             f"- Files found: **{len(docs)}**",
             f"- Current documents: **{sum(d['status'] == 'current' for d in docs)}**",
             f"- Outdated versions: **{sum(d['status'] == 'outdated' for d in docs)}**",
             f"- Exact duplicates removed: **{len(report['exact_duplicates'])}**",
             f"- Chunks indexed: **{n_chunks}**\n",
             "## Documents\n", "| File | Department | Date | Status |", "|---|---|---|---|"]
    for d in sorted(docs, key=lambda x: (x["department"], x["source"])):
        lines.append(f"| {d['source']} | {d['department']} | {d['date'] or '?'} | {d['status']} |")
    lines.append("\n## Exact duplicates (removed)\n")
    lines += [f"- `{a}` is a copy of `{b}`" for a, b in report["exact_duplicates"]] or ["- none"]
    lines.append("\n## Versions of the same document (newest kept as current)\n")
    for fam in report["families"]:
        lines.append("- " + " → ".join(f"`{s}` ({dt or '?'}, v{v}, {st})" for s, dt, v, st in fam))
    if not report["families"]:
        lines.append("- none")
    lines.append("\n## Near-duplicate content (>60% overlap)\n")
    lines += [f"- `{a}` ≈ `{b}` (similarity {s})" for a, b, s in report["near_duplicates"]] or ["- none"]
    lines.append("\n## Documents whose text says they may be outdated\n")
    lines += [f"- `{s}`" for s in report["flagged_outdated_text"]] or ["- none"]
    lines.append("\n## Outdated by date (much older than the rest)\n")
    lines += [f"- `{s}`" for s in report["old_by_date"]] or ["- none"]
    lines.append("\n## SKU conflicts (same code in several files)\n")
    lines += [f"- `{s}` → {', '.join(f'`{x}`' for x in files)}" for s, files in report["sku_conflicts"].items()] or ["- none"]
    Path(config.REPORT_PATH).write_text("\n".join(lines), encoding="utf-8")


#Main Execution
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="analyse only, no embeddings / API calls")
    args = ap.parse_args()

    root = Path(config.DOCS_DIR)
    files = sorted(f for f in root.rglob("*") if f.is_file() and not f.name.startswith("."))
    print(f"📁 Found {len(files)} files in '{root}'")

    docs = []
    for f in files:
        text = read_file(f)
        if not text.strip():
            continue
        dt = extract_date(f.stem, text)
        docs.append({
            "source": str(f.relative_to(root)), "text": text, "title": extract_title(f, text),
            "department": extract_department(f, text), "date": dt.isoformat() if dt else None,
            "version": extract_version(f.stem, text), "mtime": f.stat().st_mtime,
            "family": family_key(f.stem), "status": "current",
        })

    report = analyse(docs)
    chunks = make_chunks(docs)
    write_report(docs, report, len(chunks))
    Path(config.CATALOG_PATH).write_text(json.dumps(
        [{k: v for k, v in d.items() if k != "text"} for d in docs], indent=2, default=str), encoding="utf-8")

    print(f"🧹 {len(report['exact_duplicates'])} exact duplicates, "
          f"{sum(d['status'] == 'outdated' for d in docs)} outdated versions")
    print(f"✂️  {len(chunks)} chunks  |  📝 report saved to {config.REPORT_PATH}")

   # 1. Connect (reads PINECONE_API_KEY from .env) and create the index once
    pc = Pinecone()
    if not pc.has_index(config.PINECONE_INDEX):
        print(f"📦 Creating Pinecone index '{config.PINECONE_INDEX}'...")
        pc.create_index(
            name=config.PINECONE_INDEX,
            dimension=config.EMBEDDING_DIMENSIONS,   # 512, must match the embeddings
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(config.PINECONE_INDEX)

    # 2. Embed with LangChain + upload in batches of 100
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, dimensions=config.EMBEDDING_DIMENSIONS)
    print(f"🧠 Embedding + uploading {len(chunks)} chunks to Pinecone...")
    batch = 100
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        vectors = embeddings.embed_documents([c.page_content for c in part])
        records = []
        for c, v in zip(part, vectors):
            md = {**c.metadata, "text": c.page_content}
            skus = sorted(set(re.findall(SKU_RE, c.page_content)))
            if skus:
                md["skus"] = skus              # exact product codes on this card
            records.append({
                "id": hashlib.md5(f"{c.metadata['source']}#{c.metadata['chunk']}".encode()).hexdigest(),
                "values": v,
                "metadata": md,
            })
        index.upsert(vectors=records, show_progress=False)

    print(f"✅ Done! Vectors in Pinecone: {index.describe_index_stats()['total_vector_count']}")
if __name__ == "__main__":
    main()