# 🏬 IronStore Enterprise AI Assistant

🔗 **Live demo:** https://ironstore-assistant.vercel.app

An AI-powered internal assistant that helps IronStore employees find company knowledge through natural language.
It answers questions about **policies, procedures and products** using **only** IronStore's internal documents.
It cites its sources, prefers current information, flags conflicting data, and says clearly when it cannot answer instead of guessing.

---

## 📌 Project brief

Build an AI assistant that helps employees quickly access internal company knowledge through a conversational interface.
The assistant should answer questions, explain internal procedures and help with everyday work tasks using the organization's
proprietary information, give clear, accurate and trustworthy responses, and acknowledge when it cannot confidently answer.
The provided documents may contain **inconsistencies, duplicated content and outdated information**.

---

## 🛠️ Tech stack

| Part | Tool |
|---|---|
| Framework | LangChain |
| LLMs (model routing) | OpenAI `gpt-5.4-mini` (easy questions + routing) / `gpt-5.4` (hard questions) |
| Embeddings | OpenAI `text-embedding-3-small` (512 dimensions) |
| Vector database | Pinecone (serverless) |
| NLP / data processing | Python, regex, pypdf, MD5 hashing, Jaccard similarity |
| Backend API | FastAPI |
| Frontend | HTML + JavaScript |
| Deployment | Vercel |

---

## 📁 Project structure

```
├── documents/                    # Company PDFs (provided as documents.zip, not committed unzipped)
├── 01_data_exploration.ipynb     # Step 1: inspection of the documents
├── config.py                     # All settings (models, chunk size, thresholds, index name)
├── ingest.py                     # Load → clean → metadata → duplicates/outdated → chunk → embed → Pinecone
├── rag.py                        # Routing → retrieval → guardrails → answer generation
├── app.py                        # FastAPI app (API + chat page), deployed on Vercel
├── templates/index.html          # Chat interface
├── evaluate.py                   # Automatic evaluation
├── eval_questions.csv            # Test questions with expected keywords
├── data_report.md                # Generated data-quality report
├── requirements.txt              # Packages for Vercel (runtime)
└── requirements-dev.txt          # Extra packages for local data processing and notebooks
```

---

## 🔍 1. Data exploration

Done in `01_data_exploration.ipynb`.

**What the data contains:**
- **83 PDF documents (~2,300 pages)**
- **20 internal policies** across 7 areas: HR, IT, Finance, Sales, Legal, Deliveries, Customer Support
- **63 product catalogs** with about **1,500 SKUs** (prices, specifications, availability, warehouse locations)
- Every policy has a structured header: *Owning area, Version, Last reviewed*

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
PDFs → load → NLP cleaning → metadata (department, date, version, SKUs)
     → duplicate & outdated detection → chunking → embeddings → Pinecone

QUESTION ANSWERING  (app.py + rag.py, deployed on Vercel)
Browser → FastAPI /api/chat
        → rewrite follow-up questions (uses chat history)
        → ROUTER (cheap model): difficulty, department, on-topic?
        → hybrid retrieval: exact SKU match + semantic search (department first, then all)
        → guardrail: low similarity → honest "I couldn't find this"
        → cheap or smart model → answer with sources + confidence
```

The server is stateless (Vercel functions), so the **browser keeps the chat history** and sends it with every question.

---

## 🧹 3. How messy data is handled

| Problem | Solution |
|---|---|
| Broken text | Regex cleaning: remove HTML tags, fix known broken words (Wrocław, Poznań, Łódź, CO2), normalize quotes and bullets, tidy whitespace |
| Department labels | Read from the `Owning area:` header, then the folder name, then keyword scoring |
| Dates and versions | Regex for several date formats, with priority: *Last reviewed / Effective date* → file name → newest date in the text |
| Exact duplicates | MD5 fingerprints of normalized text, for whole documents and for individual chunks |
| Versions of one document | Grouped by cleaned file name or >60% Jaccard similarity; the newest becomes `current`, the others `outdated` |
| Outdated documents | A review date 2+ years older than the rest of the collection → marked `outdated` |
| SKU conflicts | Detected at ingestion (`data_report.md`); the prompt tells the model to show both products and recommend confirmation |
| Discontinued products | The prompt tells the model to say so and name the replacement product |
| Exact product codes | Hybrid search: SKUs stored as Pinecone metadata for exact lookup, combined with semantic search |

Outdated chunks are kept in the index but labeled, so the assistant can say *"this changed"* while answering from current documents.

---

## 🛡️ 4. Trustworthiness

- Answers **only** from retrieved context, never invents policies, numbers or names
- **Source citations** for every fact
- **Similarity threshold:** no relevant documents → honest *"I couldn't find this in the IronStore documents"* + who to contact
- **Partial answers** explain what the documents *do* say and where to find the rest
- **Off-topic questions** are politely declined
- **Confidence level** (High / Medium / Low) on every answer
- Temperature 0 for consistent answers across employees

---

## 🔀 5. Model routing

A cheap model fills in a structured routing form for every question:

| Field | Use |
|---|---|
| `difficulty` | easy → `gpt-5.4-mini` (fast, cheap) · hard → `gpt-5.4` (more accurate) |
| `department` | Search that department's documents first, fall back to all documents |
| `is_company_question` | Decline unrelated questions (e.g. world capitals) |

This keeps costs low without losing quality on complex questions.

---

## 📝 6. Evaluation

`evaluate.py` runs the questions in `eval_questions.csv` and checks each answer for expected keywords
(or for a refusal, on out-of-scope questions). Results are saved to `eval_results.csv`.

**Final score: 8/8 (100%)**, covering policy procedures, product prices, SKU conflicts and out-of-scope questions.

**Lessons learned:**
- Two tests failed at first. Checking the source documents showed they contain **no fixed number of vacation days**
  and **no password-reset steps**. The assistant correctly **did not invent** them, so the tests were fixed, not the bot.
- **Prompt engineering:** the first prompt made the assistant say *"I couldn't find this"* even when it had a partial answer.
  Separating *no answer* from *partial answer* in the rules made responses clearer.
- **Exact codes need keyword search:** semantic search alone could not find `IS-BED-001`; adding exact SKU lookup fixed it.

---

## ▶️ 7. Run locally

```bash
# 1. Install
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements-dev.txt

# 2. Create .env
#    OPENAI_API_KEY=...
#    PINECONE_API_KEY=...

# 3. Unzip documents/documents.zip into documents/

# 4. Analyse the data (no API calls) → data_report.md
python ingest.py --dry-run

# 5. Embed and upload to Pinecone
python ingest.py

# 6. Start the app → http://127.0.0.1:8000
uvicorn app:app --reload

# 7. Run the evaluation
python evaluate.py
```

## 🚀 8. Deployment (Vercel)

1. Push the repository to GitHub (`.env`, `venv/` and the unzipped documents are in `.gitignore`)
2. Import the repository in Vercel: Root Directory `./`, preset **FastAPI**
3. Add the environment variables `OPENAI_API_KEY` and `PINECONE_API_KEY`
4. Deploy. Every push to GitHub redeploys automatically

Only the lightweight runtime packages are in `requirements.txt`. The heavy document processing runs once locally,
and the vectors live in Pinecone, so no company data is stored in the repository or the deployment.

---

## ⚠️ 9. Limitations & future work

- **Synonyms** ("vacation" vs "annual leave") can weaken retrieval → add query expansion
- Add a **re-ranker** to improve the order of retrieved chunks
- **Role-based access** (e.g. HR-only documents for HR staff)
- **User feedback** buttons (👍 / 👎) and monitoring with LangSmith
- A **larger evaluation set** covering every department
- Ask the data owners to fix the SKU conflicts and archive the 2021 catalog at the source
