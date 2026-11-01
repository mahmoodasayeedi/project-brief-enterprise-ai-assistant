# 🏬 IronStore Enterprise AI Assistant

🔗 **Live demo:** https://ironstore-assistant.vercel.app

An AI-powered internal assistant that helps IronStore employees find company knowledge through natural language.
It answers questions about **policies, procedures and products** using **only** IronStore's internal documents.
Answers stream in live, cite the exact **document, page and section** with a link to the PDF, prefer current information,
flag conflicting data, and say clearly when the assistant cannot answer instead of guessing.

---

## 📌 Project brief

Build an AI assistant that helps employees quickly access internal company knowledge through a conversational interface.
The assistant should answer questions, explain internal procedures and help with everyday work tasks using the organization's
proprietary information, give clear, accurate and trustworthy responses, and acknowledge when it cannot confidently answer.
The provided documents may contain **inconsistencies, duplicated content and outdated information**.

**Bonus tasks:** show the source of each answer (document, part of the document, and a link), build and test an API,
build a frontend app connected to it, and deploy.

---

## ✅ Requirements checklist

| Requirement | Status | How |
|---|---|---|
| Conversational interface | ✅ | Chat web app with conversation memory |
| Answer questions & explain procedures | ✅ | RAG over 83 internal documents, numbered steps for procedures |
| Clear, accurate, trustworthy answers | ✅ | Grounded answers, citations, confidence level |
| Acknowledge when it can't answer | ✅ | Similarity threshold + honest "I couldn't find this" |
| Handle inconsistencies, duplicates, outdated info | ✅ | Data-quality pipeline + prompt rules (see section 4) |
| **Bonus:** source document, part and link | ✅ | File + page + section, with a link that opens the PDF on that page |
| **Bonus:** API + tests | ✅ | FastAPI (normal + streaming endpoints), `pytest` tests, Swagger UI |
| **Bonus:** frontend app | ✅ | HTML/JavaScript chat with live streaming |
| **Bonus:** deployment | ✅ | Vercel |

---

## 🛠️ Tech stack & choices

| Part | Choice | Why |
|---|---|---|
| Framework | **LangChain v1** (`langchain-core`, `langchain-openai`) | Modern API: `with_structured_output`, `.stream()` |
| LLMs | OpenAI **`gpt-5.4-mini`** + **`gpt-5.4`** with **model routing** | Cheap model for easy questions and routing, capable model only when needed |
| Embeddings | OpenAI **`text-embedding-3-small`**, 512 dimensions | Very cheap; shorter vectors keep the index small with little quality loss |
| Vector database | **Pinecone** (serverless) | Hosted, free tier, metadata filters (department, SKU) |
| NLP / data processing | Python, regex, `pypdf`, MD5 hashing, Jaccard similarity | Cleaning, metadata extraction, duplicate and outdated detection |
| Backend API | **FastAPI** | Fast, typed, auto-generated docs, streaming support |
| Frontend | HTML + JavaScript (marked + DOMPurify) | Lightweight, no build step, works on Vercel |
| Deployment | **Vercel** | Zero-config FastAPI hosting with streaming |

---

## 📁 Project structure

```
├── documents/                    # Company PDFs (provided as documents.zip; the unzipped folder is not committed)
├── public/files/                 # Copies of the PDFs, served so answers can link to them
├── templates/index.html          # Chat interface (streaming, source links)
├── 01_data_exploration.ipynb     # Step 1: inspection of the documents
├── config.py                     # All settings (models, chunk size, thresholds, index name)
├── ingest.py                     # Load → clean → metadata → duplicates/outdated → chunk → embed → Pinecone
├── rag.py                        # Routing → retrieval → guardrails → streamed answer generation
├── app.py                        # FastAPI app: chat page, API endpoints, PDF files
├── test_api.py                   # API tests (pytest)
├── evaluate.py                   # Answer-quality evaluation
├── eval_questions.csv            # Evaluation questions with expected keywords
├── data_report.md                # Generated data-quality report
├── requirements.txt              # Runtime packages (installed by Vercel)
└── requirements-dev.txt          # Extra packages for local data processing, notebooks and tests
```

---

## 🔍 1. Data exploration

Done in `01_data_exploration.ipynb`.

**What the data contains:**
- **83 PDF documents (~2,300 pages, 3.8 MB)**
- **20 internal policies** across 7 areas: HR, IT, Finance, Sales, Legal, Deliveries, Customer Support
- **63 product catalogs** with about **1,500 SKUs** (prices, specifications, availability, warehouse locations)
- Every policy has a structured header: *Owning area, Version, Last reviewed*, and numbered sections (e.g. `4.1 Entitlement`)

**Data quality problems found:**

| Problem | Example |
|---|---|
| Outdated document | `IronStore_Old_Product_Catalog_2021.pdf` was last reviewed in Dec 2021; all other documents in Aug 2026 |
| SKU conflicts | `IS-BED-001` to `IS-BED-004` are **different products** in the Bedding & Linens and Home Essentials catalogs |
| Overlapping content | Home Essentials repeats products from other catalogs |
| Inconsistent formats | `Last reviewed in August 2026` vs `Last reviewed: August 2026`; different SKU layouts |
| Text noise | HTML leftovers (`<div style="page-break-after...">`), broken characters (`Wroc■aw`, `CO■`), hidden bullet characters |
| Product-level outdated info | Items marked *Discontinued* / *Replaced by* inside current catalogs |
| Junk files | `__MACOSX/` and `.DS_Store` files in the archive |

---

## 🧠 2. Architecture

```
INGESTION  (ingest.py, run once locally)
PDFs → read page by page → NLP cleaning → metadata (department, date, version, page, section, SKUs)
     → duplicate & outdated detection → chunking per page → embeddings → Pinecone

QUESTION ANSWERING  (app.py + rag.py, deployed on Vercel)
Browser → FastAPI /api/chat/stream
        → rewrite follow-up questions (uses chat history)
        → ROUTER (cheap model): difficulty, department, on-topic?
        → hybrid retrieval: exact SKU match + semantic search (department first, then all)
        → guardrail: low similarity → honest "I couldn't find this"
        → cheap or smart model → answer streamed token by token
        → sources with page, section and PDF link
```

The server is stateless (Vercel functions), so the **browser keeps the chat history** and sends it with every question.

---

## 🧹 3. Ingestion pipeline (`ingest.py`)

| Step | What happens |
|---|---|
| Load | PDFs are read **page by page** with `pypdf`, so every chunk knows its page number |
| Clean | Regex cleaning: remove HTML tags, fix known broken words (Wrocław, Poznań, Łódź, CO2), normalize quotes and bullets, tidy whitespace |
| Metadata | Department from the `Owning area:` header (then folder name, then keywords); dates in several formats with priority *Last reviewed / Effective date* → file name → text; version numbers |
| Sections | Headings such as `4.2 Request and Approval` or `Page 3 — Running and Training` are detected, and every chunk is labeled with its section |
| Quality checks | Duplicates, versions, outdated documents and SKU conflicts (see section 4) → `data_report.md` |
| Chunk | `RecursiveCharacterTextSplitter` (1,000 characters, 150 overlap) per page; each chunk gets a header with title, department, date, page and section |
| Embed & store | `text-embedding-3-small` (512 dims) → Pinecone, with metadata (department, status, page, section, SKUs) and stable IDs so re-running updates instead of duplicating |

Run `python ingest.py --dry-run` to analyse the data without any API calls.

---

## 🧽 4. How messy data is handled

| Problem | Solution |
|---|---|
| Broken text | Regex cleaning before chunking |
| Exact duplicates | MD5 fingerprints of normalized text, for whole documents and individual chunks |
| Versions of one document | Grouped by cleaned file name or >60% Jaccard similarity; the newest becomes `current`, the others `outdated` |
| Outdated documents | A review date 2+ years older than the rest of the collection → marked `outdated` |
| SKU conflicts | Detected at ingestion; the prompt tells the model to show both products and recommend confirmation |
| Discontinued products | The prompt tells the model to say so and name the replacement product |
| Exact product codes | **Hybrid search**: SKUs stored as Pinecone metadata for exact lookup, combined with semantic search |

Outdated chunks stay in the index but are labeled, so the assistant can say *"this changed"* while answering from current documents.

---

## 🛡️ 5. Trustworthiness & sources

- Answers **only** from retrieved context, never invents policies, numbers or names
- Every fact cites its source with the page, e.g. `[IronStore_Leave_and_Absence_Policy.pdf, p. 2]`
- Citations become **clickable links that open the PDF on the cited page**
- A **Sources** panel lists each document with **page, section** (e.g. *4.2 Request and Approval*), date, current/outdated status and relevance score
- **Similarity threshold:** no relevant documents → honest *"I couldn't find this in the IronStore documents"* + who to contact
- **Partial answers** explain what the documents *do* say and where to find the rest
- **Off-topic questions** are politely declined
- **Confidence level** (High / Medium / Low) on every answer
- Temperature 0 for consistent answers across employees

---

## 🔀 6. Model routing

A cheap model fills in a structured routing form (`with_structured_output`) for every question:

| Field | Use |
|---|---|
| `difficulty` | easy → `gpt-5.4-mini` (fast, cheap) · hard → `gpt-5.4` (more accurate) |
| `department` | Search that department's documents first, fall back to all documents |
| `is_company_question` | Decline unrelated questions (e.g. world capitals) |

This keeps costs low without losing quality on complex questions. The chat shows the routing decision as badges on every answer.

---

## ⚡ 7. Streaming

Answers appear **token by token**, like ChatGPT. Streaming works end to end:

| Layer | How it streams |
|---|---|
| OpenAI → `rag.py` | LangChain `llm.stream()` |
| `rag.py` → `app.py` | `ask_stream()` is a Python **generator** that `yield`s a `meta` event (routing + sources), then `token` events |
| `app.py` → browser | FastAPI `StreamingResponse` sending **NDJSON** (one JSON object per line) |
| Browser | `fetch` + `response.body.getReader()` renders the growing answer |

`ask()` collects the same stream into one response, used by the non-streaming endpoint and the evaluation.

---

## 🌐 8. API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Chat web app |
| `/api/health` | GET | Health check → `{"status": "ok"}` |
| `/api/chat` | POST | Full answer as one JSON (`answer`, `sources`, `route`, `model`) |
| `/api/chat/stream` | POST | Streamed answer (NDJSON: `meta`, then `token` events) |
| `/files/<name>.pdf` | GET | Source documents (used by the citation links) |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

**Request body** (`/api/chat` and `/api/chat/stream`):
```json
{
  "question": "How do I request annual leave?",
  "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
}
```

**Example source in the response:**
```json
{
  "file": "IronStore_Leave_and_Absence_Policy.pdf",
  "page": 2,
  "section": "4.2 Request and Approval",
  "url": "/files/IronStore_Leave_and_Absence_Policy.pdf#page=2",
  "status": "current",
  "score": 0.71
}
```

---

## 🧪 9. Testing & evaluation

### API tests (`test_api.py`, run with `pytest -v`)
**6/6 passed:** health check, chat page loads, empty questions rejected, answers include sources with page and a working PDF link,
off-topic questions declined, streaming sends `meta` then `token` events.

### Answer-quality evaluation (`evaluate.py`)
**Score: 8/8 (100%)**, covering policy procedures, product prices, SKU conflicts and out-of-scope questions.
Results are saved to `eval_results.csv`.

**Lessons learned:**
- Two tests failed at first. Checking the source documents showed they contain **no fixed number of vacation days**
  and **no password-reset steps**. The assistant correctly **did not invent** them, so the tests were fixed, not the bot.
- **Prompt engineering:** the first prompt made the assistant say *"I couldn't find this"* even when it had a partial answer.
  Separating *no answer* from *partial answer* in the rules made responses clearer.
- **Exact codes need keyword search:** semantic search alone could not find `IS-BED-001`; adding exact SKU lookup fixed it.
- **Streaming is end to end:** `streaming=True` on the model is not enough on its own; every layer (LangChain, generator,
  FastAPI, browser) has to pass the tokens on.

---

## ▶️ 10. Run locally

```bash
# 1. Install
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements-dev.txt

# 2. Create .env
#    OPENAI_API_KEY=...
#    PINECONE_API_KEY=...

# 3. Unzip documents/documents.zip into documents/, then copy the PDFs for the links (PowerShell):
#    New-Item -ItemType Directory -Force public\files
#    Get-ChildItem documents -Recurse -Filter *.pdf | Where-Object { $_.FullName -notmatch "__MACOSX" } | Copy-Item -Destination public\files

# 4. Analyse the data (no API calls) → data_report.md
python ingest.py --dry-run

# 5. Embed and upload to Pinecone
python ingest.py

# 6. Start the app → http://127.0.0.1:8000   (API docs: http://127.0.0.1:8000/docs)
uvicorn app:app --reload

# 7. Test
pytest -v
python evaluate.py
```

## 🚀 11. Deployment (Vercel)

1. Push the repository to GitHub (`.env`, `venv/` and the unzipped documents are in `.gitignore`)
2. Import the repository in Vercel: Root Directory `./`, preset **FastAPI**
3. Add the environment variables `OPENAI_API_KEY` and `PINECONE_API_KEY`
4. Deploy. Every push to GitHub redeploys automatically

Only lightweight runtime packages are in `requirements.txt`. The heavy document processing runs once locally,
and the vectors live in Pinecone. The `public/` folder is served automatically, which makes the PDF links work.

---

## ⚠️ 12. Limitations & future work

- **Document access:** for this prototype the PDFs are public in `public/files` (the data is synthetic and was already in the repo).
  In production they would stay in access-controlled storage (e.g. SharePoint or S3 with signed URLs) behind a login
- **Synonyms** ("vacation" vs "annual leave") can weaken retrieval → add query expansion
- Add a **re-ranker** to improve the order of retrieved chunks
- **Role-based access** (e.g. HR-only documents for HR staff)
- **User feedback** buttons (👍 / 👎) and monitoring with LangSmith
- A **larger evaluation set** covering every department
- Ask the data owners to fix the SKU conflicts and archive the 2021 catalog at the source
