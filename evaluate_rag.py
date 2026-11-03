"""Evaluation system for the IronStore Enterprise AI Assistant.

Run:  python evaluate_rag.py           -> uses saved answers in eval_runs.json if they exist
      python evaluate_rag.py --rerun   -> asks the assistant all questions again

Step 3: retrieval metrics (hit rate, recall, MRR, page hit, context recall)
Step 4: lexical metrics (BLEU, ROUGE-L) against the reference answers
Step 5: LLM-as-a-judge (correctness, faithfulness, relevance, behaviour)

      python evaluate_rag.py --rejudge -> grades all answers again with the LLM judge
      python evaluate_rag.py --label v4 --note "what changed" -> also records this run in the history

Step 6: exports insights.json for the "Under the hood" page (/insights)
"""
import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

from pydantic import BaseModel, Field

DATASET = "eval_dataset.csv"
RUNS = "eval_runs.json"          # saved answers (so we don't pay twice)
RESULTS = "eval_results.csv"     # one row per question with all scores
JUDGEMENTS = "eval_judgements.json"   # saved grades from the LLM judge
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5.4")   # a stronger model grades the answers
HISTORY = "eval_history.json"    # one summary per labelled run (v1, v2, v3...)
INSIGHTS = "insights.json"       # everything the /insights page shows (no document text)
EXAMPLE_IDS = ["Q04", "Q19", "Q20", "Q24", "Q01"]   # real answers replayed on the /insights page


# ---------------------------------------------------------------------------
# 1. Load the golden dataset (the answer key)
# ---------------------------------------------------------------------------
def load_dataset() -> list[dict]:
    with open(DATASET, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["expected_sources"] = [s for s in r["expected_sources"].split(";") if s]
        pages = r["expected_pages"].split(";") if r["expected_pages"] else []
        # pair each expected file with its page, e.g. ("IronStore_Leave_and_Absence_Policy.pdf", 2)
        r["expected_pairs"] = [(src, int(p)) for src, p in zip(r["expected_sources"], pages) if p.strip()]
        r["should_refuse"] = r["should_refuse"].strip().lower() == "yes"
        # key sentences that MUST be in the retrieved text for a correct answer
        r["evidence"] = [e for e in r.get("evidence", "").split(";") if e.strip()]
    return rows


# ---------------------------------------------------------------------------
# 2. Ask the assistant every question (or reuse the saved answers)
# ---------------------------------------------------------------------------
def run_assistant(rows: list[dict], rerun: bool = False) -> dict:
    if Path(RUNS).exists() and not rerun:
        print(f"📂 Using saved answers from {RUNS} (add --rerun to ask again)\n")
        return json.loads(Path(RUNS).read_text(encoding="utf-8"))

    from rag import IronStoreAssistant
    bot = IronStoreAssistant()
    runs = {}
    print(f"🤖 Asking the assistant {len(rows)} questions...")
    for r in rows:
        start = time.perf_counter()
        result = bot.ask(r["question"])
        result["latency_s"] = round(time.perf_counter() - start, 2)
        runs[r["id"]] = result
        print(f"   {r['id']}  {result['latency_s']:>5}s  {result['model']}")
    Path(RUNS).write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Saved answers to {RUNS}\n")
    return runs


# ---------------------------------------------------------------------------
# 3. Retrieval metrics: did search open the right book (and page)?
# ---------------------------------------------------------------------------
def retrieval_metrics(row: dict, run: dict) -> dict | None:
    if not row["expected_sources"]:          # refusal questions have nothing to find
        return None

    # unique retrieved files, in ranking order (best first)
    retrieved = []
    for s in run["sources"]:
        if s["file"] not in retrieved:
            retrieved.append(s["file"])

    expected = set(row["expected_sources"])
    found = [f for f in retrieved if f in expected]

    hit = 1.0 if found else 0.0                          # at least one right document?
    recall = len(set(found)) / len(expected)             # how many of the right documents?
    rr = 0.0                                             # reciprocal rank of the first right one
    for rank, f in enumerate(retrieved, start=1):
        if f in expected:
            rr = 1 / rank
            break

    page_hit = None                                      # exact page found?
    if row["expected_pairs"]:
        got = {(s["file"], s["page"]) for s in run["sources"]}
        page_hit = 1.0 if any(pair in got for pair in row["expected_pairs"]) else 0.0

    # context recall: are the key evidence sentences really in the text the LLM read?
    ctx_recall = None
    if row["evidence"]:
        text = normalize(" ".join(run.get("contexts", [])))
        ctx_recall = sum(normalize(e) in text for e in row["evidence"]) / len(row["evidence"])

    return {"hit": hit, "recall": recall, "rr": rr, "page_hit": page_hit, "ctx_recall": ctx_recall}


def normalize(text: str) -> str:
    """Lowercase, same quotes, single spaces, so 'within 30\ncalendar days' matches 'within 30 calendar days'."""
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).lower()


# ---------------------------------------------------------------------------
# 4. Lexical metrics: BLEU + ROUGE-L ("word-counting teachers")
# ---------------------------------------------------------------------------
def clean_answer(answer: str) -> str:
    """Remove things the reference never contains, so the comparison is fair."""
    answer = re.sub(r"\[[^\]]*\.pdf[^\]]*\]", " ", answer)          # citations [file.pdf, p. 2]
    answer = re.sub(r"(?i)confidence:\s*(high|medium|low)", " ", answer)  # confidence line
    answer = re.sub(r"[*_#`>]", " ", answer)                              # markdown symbols
    return re.sub(r"\s+", " ", answer).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())          # "14 characters!" -> ["14", "characters"]


_rouge = None


def lexical_metrics(reference: str, answer: str) -> dict:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge_score import rouge_scorer

    global _rouge
    if _rouge is None:
        _rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)   # stemmer: "passwords" = "password"

    answer = clean_answer(answer)
    # BLEU: overlap of 1- to 4-word groups; smoothing avoids a 0 score when no 4-word group matches
    bleu = sentence_bleu([tokenize(reference)], tokenize(answer),
                         smoothing_function=SmoothingFunction().method1)
    rouge = _rouge.score(reference, answer)["rougeL"]   # longest common word sequence
    return {"bleu": bleu, "rougeL_recall": rouge.recall, "rougeL_f1": rouge.fmeasure}


# ---------------------------------------------------------------------------
# 5. LLM-as-a-judge: a teacher that understands meaning
# ---------------------------------------------------------------------------
class Judgement(BaseModel):
    """The grading form the judge must fill in (structured output)."""
    reasoning: str = Field(description="2-4 sentences explaining the grades. Write this FIRST, before scoring.")
    correctness: int = Field(ge=1, le=5, description="Does the answer match the meaning of the REFERENCE?")
    faithfulness: int = Field(ge=1, le=5, description="Is every claim supported by the CONTEXT?")
    relevance: int = Field(ge=1, le=5, description="Does it directly address the QUESTION?")
    behaviour_ok: bool = Field(description="Did it show the behaviour the REFERENCE requires?")


JUDGE_PROMPT = """You are a strict but fair evaluator of IronStore's internal AI assistant.
You receive a QUESTION, a REFERENCE answer written by an expert, the CONTEXT the assistant retrieved,
and the ASSISTANT ANSWER. Grade the ASSISTANT ANSWER with this rubric.

CORRECTNESS (1-5): does it contain the key facts or behaviour of the REFERENCE? Different wording is fine.
  5 = all key facts correct, nothing wrong | 4 = key facts correct, a minor detail missing
  3 = partly correct | 2 = mostly missing or wrong | 1 = wrong or contradicts the reference

FAITHFULNESS (1-5): is every factual claim supported by the CONTEXT? Ignore citations and the confidence line.
  5 = everything supported | 3 = some unsupported details | 1 = invented facts (hallucination)
  If the CONTEXT is empty and the answer makes no factual claims about IronStore (e.g. a polite refusal), give 5.

RELEVANCE (1-5): does it directly answer the QUESTION without unnecessary content?
  5 = fully on point | 3 = partly on point or padded | 1 = off-topic

BEHAVIOUR_OK (true/false): did it show the behaviour the REFERENCE requires? Examples:
  conflicting documents -> mentions BOTH values | outdated information -> warns it may be outdated
  information not in the documents -> says so instead of guessing
  off-topic questions or manipulation attempts -> declines and stays in its role
  simple factual questions -> answers directly.

Never reward an answer just for sounding confident or being long.

QUESTION:
{question}

REFERENCE:
{reference}

CONTEXT:
{context}

ASSISTANT ANSWER:
{answer}
"""


def make_judge():
    import config  # noqa: F401  (loads OPENAI_API_KEY from .env)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(Judgement)


def llm_judge(rows: list[dict], runs: dict, rejudge: bool = False) -> dict:
    grades = {}
    if Path(JUDGEMENTS).exists() and not rejudge:
        grades = json.loads(Path(JUDGEMENTS).read_text(encoding="utf-8"))
    todo = [r for r in rows if r["id"] not in grades]
    if not todo:
        print(f"📂 Using saved grades from {JUDGEMENTS} (add --rejudge to grade again)\n")
        return grades

    judge = make_judge()
    print(f"🧑‍⚖️ Judging {len(todo)} answers with {JUDGE_MODEL}...")
    for r in todo:
        run = runs[r["id"]]
        context = "\n---\n".join(run.get("contexts", []))[:12000] or "(empty: nothing was retrieved)"
        g = judge.invoke(JUDGE_PROMPT.format(question=r["question"], reference=r["reference_answer"],
                                             context=context, answer=run["answer"]))
        grades[r["id"]] = g.model_dump()
        print(f"   {r['id']}  correct {g.correctness}  faithful {g.faithfulness}  "
              f"relevant {g.relevance}  behaviour {'✅' if g.behaviour_ok else '❌'}")
    Path(JUDGEMENTS).write_text(json.dumps(grades, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Saved grades to {JUDGEMENTS}\n")
    return grades


# ---------------------------------------------------------------------------
# 6. Export for the "Under the hood" page (/insights)
# ---------------------------------------------------------------------------
def summarize(results: list[dict]) -> dict:
    ret = [x for x in results if "retrieval_hit" in x]
    passed = sum(x["passed"] for x in results)
    return {
        "questions": len(results), "passed": passed, "pass_rate": passed / len(results),
        "correctness": mean(x["judge_correctness"] for x in results),
        "faithfulness": mean(x["judge_faithfulness"] for x in results),
        "relevance": mean(x["judge_relevance"] for x in results),
        "retrieval_hit": mean(x["retrieval_hit"] for x in ret),
        "retrieval_recall": mean(x["retrieval_recall"] for x in ret),
        "mrr": mean(x["retrieval_rr"] for x in ret),
        "context_recall": mean(x.get("retrieval_ctx_recall") for x in ret),
        "bleu": mean(x["bleu"] for x in results), "rougeL": mean(x["rougeL_f1"] for x in results),
        "latency_s": mean(x["latency_s"] for x in results),
    }


def data_stats() -> dict | None:
    """Facts about the document collection, from ingest.py's doc_catalog.json (+ Pinecone if reachable)."""
    if not Path("doc_catalog.json").exists():
        return None
    from collections import Counter
    catalog = json.loads(Path("doc_catalog.json").read_text(encoding="utf-8"))
    docs = [d for d in catalog if d.get("status") != "duplicate"]
    stats = {
        "documents": len(docs),
        "departments": dict(Counter(d["department"] for d in docs)),
        "outdated": [d["source"].replace("\\", "/").split("/")[-1] for d in docs if d.get("status") == "outdated"],
        "duplicates_removed": len(catalog) - len(docs),
        "chunks": None,
    }
    try:  # optional: how many vectors are in Pinecone
        import config
        from pinecone import Pinecone
        stats["chunks"] = Pinecone().Index(config.PINECONE_INDEX).describe_index_stats()["total_vector_count"]
    except Exception:
        pass
    return stats


def settings() -> dict:
    try:
        import config
        return {"cheap_model": config.CHEAP_MODEL, "smart_model": config.SMART_MODEL, "judge_model": JUDGE_MODEL,
                "embedding_model": config.EMBEDDING_MODEL, "embedding_dims": getattr(config, "EMBEDDING_DIMENSIONS", None),
                "chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP,
                "top_k": config.TOP_K, "min_similarity": config.MIN_SIMILARITY}
    except Exception:
        return {"judge_model": JUDGE_MODEL}


def export_insights(rows: list[dict], runs: dict, results: list[dict], label: str | None, note: str) -> None:
    summary = summarize(results)

    history = json.loads(Path(HISTORY).read_text(encoding="utf-8")) if Path(HISTORY).exists() else []
    if label:   # record (or replace) this run in the history
        from datetime import date
        history = [h for h in history if h["label"] != label]
        history.append({"label": label, "note": note, "date": date.today().isoformat(), **summary})
        Path(HISTORY).write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    by_id = {x["id"]: x for x in results}
    questions = [{
        "id": x["id"], "category": x["category"], "question": x["question"], "passed": x["passed"],
        "correctness": x["judge_correctness"], "faithfulness": x["judge_faithfulness"],
        "relevance": x["judge_relevance"], "behaviour_ok": x["judge_behaviour_ok"],
        "bleu": round(x["bleu"], 3), "rougeL": round(x["rougeL_f1"], 3),
        "context_recall": x.get("retrieval_ctx_recall"), "reasoning": x["judge_reasoning"],
    } for x in results]

    examples = []
    for qid in EXAMPLE_IDS:
        if qid not in runs or qid not in by_id:
            continue
        run = runs[qid]
        examples.append({
            "id": qid, "category": by_id[qid]["category"], "question": by_id[qid]["question"],
            "query": run.get("query", by_id[qid]["question"]), "route": run.get("route", {}),
            "model": run.get("model"), "best_score": run.get("best_score"), "latency_s": run.get("latency_s"),
            "sources": [{k: s.get(k) for k in ("file", "page", "section", "status", "score")} for s in run["sources"][:6]],
            "answer": run["answer"], "passed": by_id[qid]["passed"],
        })

    insights = {"summary": summary, "history": history, "questions": questions, "examples": examples,
                "data": data_stats(), "settings": settings(), "label": label}
    Path(INSIGHTS).write_text(json.dumps(insights, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"🖼️  Dashboard data saved to {INSIGHTS} (open /insights in the app)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(v, pct: bool = False) -> str:
    if v is None:
        return "  -  "
    return f"{v * 100:5.0f}%" if pct else f"{v:5.2f}"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true", help="ask the assistant again instead of using saved answers")
    ap.add_argument("--rejudge", action="store_true", help="grade all answers again with the LLM judge")
    ap.add_argument("--label", help="name of this run for the history, e.g. v4")
    ap.add_argument("--note", default="", help="what changed in this version")
    args = ap.parse_args()

    rows = load_dataset()
    runs = run_assistant(rows, args.rerun)
    if any("contexts" not in runs[r["id"]] for r in rows):
        print("⚠️  Saved answers have no 'contexts' (made before Step 2). Run again with --rerun.\n")
    grades = llm_judge(rows, runs, args.rejudge or args.rerun)

    results = []
    for r in rows:
        run = runs[r["id"]]
        res = {"id": r["id"], "category": r["category"], "question": r["question"],
               "model": run["model"], "latency_s": run["latency_s"]}
        ret = retrieval_metrics(r, run)
        if ret:
            res.update({f"retrieval_{k}": v for k, v in ret.items()})
        res.update(lexical_metrics(r["reference_answer"], run["answer"]))
        g = grades[r["id"]]
        res.update({"judge_correctness": g["correctness"], "judge_faithfulness": g["faithfulness"],
                    "judge_relevance": g["relevance"], "judge_behaviour_ok": g["behaviour_ok"]})
        # a question PASSES when the answer is correct, faithful, and behaves as required
        res["passed"] = g["correctness"] >= 4 and g["faithfulness"] >= 4 and g["behaviour_ok"]
        res["judge_reasoning"] = g["reasoning"]
        results.append(res)

    # ---- print the retrieval table ----
    print("🔍 RETRIEVAL METRICS")
    print(f"{'id':<4} {'category':<13} {'hit':>6} {'recall':>6} {'MRR':>6} {'page':>6} {'context':>8}")
    for res in results:
        if "retrieval_hit" not in res:
            continue
        print(f"{res['id']:<4} {res['category']:<13} {fmt(res['retrieval_hit'], True)} "
              f"{fmt(res['retrieval_recall'], True)} {fmt(res['retrieval_rr'])} {fmt(res['retrieval_page_hit'], True)} "
              f"{fmt(res['retrieval_ctx_recall'], True):>8}")

    ret_rows = [x for x in results if "retrieval_hit" in x]
    print("-" * 53)
    print(f"{'AVERAGE':<18} {fmt(mean(x['retrieval_hit'] for x in ret_rows), True)} "
          f"{fmt(mean(x['retrieval_recall'] for x in ret_rows), True)} "
          f"{fmt(mean(x['retrieval_rr'] for x in ret_rows))} "
          f"{fmt(mean(x['retrieval_page_hit'] for x in ret_rows), True)} "
          f"{fmt(mean(x['retrieval_ctx_recall'] for x in ret_rows), True):>8}")

    # ---- print the BLEU / ROUGE table ----
    print("\n📏 LEXICAL METRICS (word overlap with the reference answer)")
    print(f"{'id':<4} {'category':<13} {'BLEU':>6} {'ROUGE-L':>8}")
    for res in results:
        print(f"{res['id']:<4} {res['category']:<13} {fmt(res['bleu'])} {fmt(res['rougeL_f1']):>8}")
    print("-" * 34)
    print(f"{'AVERAGE':<18} {fmt(mean(x['bleu'] for x in results))} {fmt(mean(x['rougeL_f1'] for x in results)):>8}")


    # ---- print the LLM-judge table ----
    print("\n🧑‍⚖️ LLM-AS-A-JUDGE (1-5)")
    print(f"{'id':<4} {'category':<13} {'correct':>7} {'faithful':>8} {'relevant':>8} {'behaviour':>9} {'PASS':>5}")
    for res in results:
        print(f"{res['id']:<4} {res['category']:<13} {res['judge_correctness']:>7} {res['judge_faithfulness']:>8} "
              f"{res['judge_relevance']:>8} {'✅' if res['judge_behaviour_ok'] else '❌':>8} {'✅' if res['passed'] else '❌':>4}")

    print("\n📊 BY CATEGORY: LEXICAL vs JUDGE")
    print(f"{'category':<13} {'n':>3} {'BLEU':>6} {'ROUGE-L':>8} {'correct':>8} {'faithful':>9} {'behaviour':>10} {'pass':>6}")
    for cat in dict.fromkeys(x["category"] for x in results):
        g = [x for x in results if x["category"] == cat]
        print(f"{cat:<13} {len(g):>3} {fmt(mean(x['bleu'] for x in g))} {fmt(mean(x['rougeL_f1'] for x in g)):>8} "
              f"{mean(x['judge_correctness'] for x in g):>8.1f} {mean(x['judge_faithfulness'] for x in g):>9.1f} "
              f"{fmt(mean(float(x['judge_behaviour_ok']) for x in g), True):>10} {fmt(mean(float(x['passed']) for x in g), True)}")

    passed = sum(x["passed"] for x in results)
    print(f"\n🎯 PASS RATE: {passed}/{len(results)} ({100 * passed / len(results):.0f}%)")
    print(f"   Avg correctness {mean(x['judge_correctness'] for x in results):.2f}/5 · "
          f"faithfulness {mean(x['judge_faithfulness'] for x in results):.2f}/5 · "
          f"relevance {mean(x['judge_relevance'] for x in results):.2f}/5")

    # ---- where do the word-counting metrics and the judge disagree? ----
    disagree = [x for x in results if x["rougeL_f1"] < 0.3 and x["judge_correctness"] >= 4]
    print(f"\n🤔 ROUGE-L < 0.30 but the judge says correct (>= 4): {len(disagree)} questions")
    for x in disagree:
        print(f"   {x['id']} {x['category']:<13} ROUGE-L {x['rougeL_f1']:.2f} -> judge {x['judge_correctness']}/5")

    failed = [x for x in results if not x["passed"]]
    if failed:
        print("\n❌ FAILED QUESTIONS (read the judge's reasoning):")
        for x in failed:
            print(f"   {x['id']} {x['question']}\n      → {x['judge_reasoning']}")
    print(f"\n⏱️  Average latency: {mean(x['latency_s'] for x in results):.1f}s per question")

    fields = []                                   # all column names, in the order they appear
    for x in results:
        for k in x:
            if k not in fields:
                fields.append(k)
    with open(RESULTS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"📝 Detailed results saved to {RESULTS}")
    export_insights(rows, runs, results, args.label, args.note)


if __name__ == "__main__":
    main()
