"""RAG"""
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from pathlib import Path
import config
import re
SKU_RE = r"\b[A-Z]{2,4}(?:-[A-Z0-9]{1,6}){2,4}\b"

DEPARTMENTS = ["HR_policies", "IT", "Finance", "Sales", "Legal_Terms",
               "Deliveries", "Customer_support", "product_catalog"]
DEPT_NAMES = DEPARTMENTS + ["General"]

class Route(BaseModel):
    """Routing decision made by the cheap model."""
    difficulty: Literal["easy", "hard"] = Field(
        description="easy = single fact lookup / short answer. hard = multi-step procedure, comparison, "
                    "reasoning across several policies, or calculations.")
    department: str = Field(description=f"Most relevant department, one of: {', '.join(DEPT_NAMES)}. "
                                        "Use product_catalog for questions about products, SKUs, prices or stock.")
    is_company_question: bool = Field(
        description="True if this is about IronStore: internal policies, procedures, work tasks, "
                    "OR IronStore products, SKUs, prices, stock, warehouses or catalogs. "
                    "False ONLY for small talk or general knowledge clearly unrelated to IronStore "
                    "(e.g. world capitals, recipes, celebrities).")


SYSTEM_PROMPT = """You are the IronStore Enterprise AI Assistant, helping employees with internal company knowledge.

RULES (follow strictly):
1. Answer ONLY using the CONTEXT below. Never invent policies, numbers, names or links.
2. If the context contains NOTHING relevant, say clearly: "I couldn't find this in the IronStore documents."
   and suggest who to contact. Do not guess.
   If the context answers only PARTIALLY (e.g. "it depends on..." or "see your PeopleHub profile"),
   do NOT use that sentence: explain what the documents DO say, then what is missing and where to find it.
3. Every fact must cite its source in square brackets, e.g. [HR/leave_policy_2025.md].
4. Documents can be outdated or conflict:
   - Prefer chunks marked STATUS: current and with the most recent DATE.
   - If an OUTDATED chunk says something different, mention briefly that the policy changed
     (e.g. "Note: the 2023 version said X; the current version says Y").
   - If two CURRENT documents disagree, say so openly and show both, recommending the employee confirm with the department.
4b. Products: if a product is marked "Discontinued", "Superseded" or "Replaced by", say so clearly
   and name the replacement product if one is given.
4c. If the same SKU appears for DIFFERENT products in different catalogs, mention the conflict
   and show both, recommending the employee confirm with the merchandising team.
5. For procedures, answer with clear numbered steps. Be concise, friendly and professional.
6. End with a confidence line: "Confidence: High / Medium / Low" based on how directly the context answers.

CONTEXT:
{context}
"""


class IronStoreAssistant:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, dimensions=config.EMBEDDING_DIMENSIONS)
        self.index = Pinecone().Index(config.PINECONE_INDEX)
        self.cheap = ChatOpenAI(model=config.CHEAP_MODEL, temperature=0)
        self.smart = ChatOpenAI(model=config.SMART_MODEL, temperature=0)
        self.router = self.cheap.with_structured_output(Route)

    # -- 1. follow-up questions -------------------------------------------------
    def standalone_question(self, question: str, history: list[dict]) -> str:
        if not history:
            return question
        convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-6:])
        prompt = ("Rewrite the user's last question so it can be understood without the conversation. "
                  "Return only the rewritten question.\n\n"
                  f"Conversation:\n{convo}\n\nLast question: {question}")
        return self.cheap.invoke(prompt).content.strip()

    # -- 3. retrieval -------------------------------------------------------------
    def retrieve(self, query: str, department: str):
        qvec = self.embeddings.embed_query(query)  # turn the question into a "meaning address" once

        def search(flt=None):
            res = self.index.query(vector=qvec, top_k=config.TOP_K + 4, include_metadata=True, filter=flt)
            out = []
            for m in res.matches:
                md = dict(m.metadata)
                text = md.pop("text", "")
                out.append((Document(page_content=text, metadata=md), m.score))
            return out
                # Exact SKU lookup (keyword search). An exact match counts as 100% relevant
        skus = sorted(set(re.findall(SKU_RE, query.upper())))
        sku_hits = [(d, 1.0) for d, s in search({"skus": {"$in": skus}})] if skus else []
        results = []
        if department in DEPARTMENTS:  # department routing first (RAG routing!)
            results = search({"department": {"$eq": department}})

        if not results or results[0][1] < config.MIN_SIMILARITY + 0.1:
            results = search()  # fall back to all departments
        
        seen = {d.page_content for d, _ in sku_hits}
        results = sku_hits + [(d, s) for d, s in results if d.page_content not in seen]
        # current docs first, then at most 2 outdated chunks (only to explain changes)
        current = [(d, s) for d, s in results if d.metadata.get("status") == "current"][: config.TOP_K]
        outdated = [(d, s) for d, s in results if d.metadata.get("status") == "outdated"][:2]
        hits = sorted(current + outdated, key=lambda x: x[1], reverse=True)
        return hits

    @staticmethod
    def format_context(hits) -> str:
        blocks = []
        for doc, score in hits:
            m = doc.metadata
            extra = f" | SUPERSEDED BY: {m['superseded_by']}" if m.get("superseded_by") else ""
            warn = f" | WARNING: {m['warning']}" if m.get("warning") else ""
            blocks.append(f"--- SOURCE: {Path(m['source']).name} | DEPT: {m['department']} | DATE: {m['date']} "
                          f"| STATUS: {m['status']}{extra}{warn} | relevance={score:.2f}\n{doc.page_content}")
        return "\n\n".join(blocks)

        # -- main entry point (STREAMING) ---------------------------------------------
    def ask_stream(self, question: str, history: list[dict] | None = None):
        """Yields events: first {"type": "meta", ...} with routing + sources,
        then many {"type": "token", "text": "..."} pieces of the answer."""
        history = history or []
        query = self.standalone_question(question, history)
        route = self.router.invoke(query)
        meta = {"type": "meta", "route": route.model_dump(), "query": query,
                "model": config.CHEAP_MODEL, "sources": [], "best_score": 0.0}

        if not route.is_company_question and route.department == "General":
            yield meta
            yield {"type": "token", "text": "I'm IronStore's internal assistant, so I can only help with company "
                                            "policies, procedures, products and work tasks. "
                                            "What would you like to know about IronStore?"}
            return

        hits = self.retrieve(query, route.department)
        best = float(hits[0][1]) if hits else 0.0
        meta["best_score"] = best

        # -- guardrail: be honest when we don't know ---------------------------
        if best < config.MIN_SIMILARITY:
            meta["model"] = "-"
            yield meta
            yield {"type": "token", "text": "I couldn't find reliable information about this in the IronStore "
                                            "documents, so I don't want to guess. Please contact the "
                                            f"**{route.department}** team or check with your manager.\n\nConfidence: Low"}
            return

        # -- model routing: easy -> cheap, hard -> smart ----------------------
        llm = self.smart if route.difficulty == "hard" else self.cheap
        messages = [SystemMessage(SYSTEM_PROMPT.format(context=self.format_context(hits)))]
        for m in history[-6:]:
            messages.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))
        messages.append(HumanMessage(question))

        sources = []
        for doc, score in hits:
            m = doc.metadata
            if m["source"] not in [s["source"] for s in sources]:
                sources.append({"source": m["source"], "department": m["department"], "date": m["date"],
                                "status": m["status"], "score": round(float(score), 2)})
        meta.update(model=llm.model_name, sources=sources)
        yield meta                                   # 1) send routing + sources first

        for chunk in llm.stream(messages):           # 2) then the answer, piece by piece
            if chunk.text:
                yield {"type": "token", "text": str(chunk.text)}

    # -- non-streaming version (used by evaluate.py) ---------------------------------
    def ask(self, question: str, history: list[dict] | None = None) -> dict:
        """Collects the whole stream into one dict, like before."""
        result, answer = {}, ""
        for event in self.ask_stream(question, history):
            if event["type"] == "meta":
                result = {k: v for k, v in event.items() if k != "type"}
            else:
                answer += event["text"]
        result["answer"] = answer
        return result

if __name__ == "__main__":  # quick terminal chat
    bot = IronStoreAssistant()
    hist = []
    print("IronStore Assistant (type 'exit' to quit)")
    while (q := input("\nYou: ").strip()).lower() not in {"exit", "quit"}:
        r = bot.ask(q, hist)
        print(f"\n🤖 [{r['model']} | {r['route']['difficulty']} | {r['route']['department']}]\n{r['answer']}")
        hist += [{"role": "user", "content": q}, {"role": "assistant", "content": r["answer"]}]

