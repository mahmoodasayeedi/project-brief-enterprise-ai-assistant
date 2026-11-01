
import csv

from rag import IronStoreAssistant

REFUSAL_WORDS = ["couldn't find", "could not find", "don't know", "not able to find", "only help with", "no information"]

bot = IronStoreAssistant()
rows = list(csv.DictReader(open("eval_questions.csv", encoding="utf-8")))
passed = 0
results = []
for row in rows:
    r = bot.ask(row["question"])
    ans = r["answer"].lower()
    if row.get("should_refuse", "").strip().lower() == "yes":
        ok = any(w in ans for w in REFUSAL_WORDS)
    else:
        kws = [k.strip().lower() for k in row["expected_keywords"].split(";") if k.strip()]
        ok = all(k in ans for k in kws)
    passed += ok
    results.append({"question": row["question"], "pass": ok, "model": r["model"],
                    "difficulty": r["route"]["difficulty"], "department": r["route"]["department"],
                    "sources": "; ".join(s["source"] for s in r["sources"]), "answer": r["answer"]})
    print(f"{'✅' if ok else '❌'} [{r['model']}] {row['question']}")

print(f"\n🎯 Score: {passed}/{len(rows)} ({100 * passed / max(1, len(rows)):.0f}%)")
with open("eval_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)
print("Detailed results saved to eval_results.csv")
