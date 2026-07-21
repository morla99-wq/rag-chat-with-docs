# eval.py
from generate import answer_question


# Write 5-10 question/expected-source pairs based on your own docs
EVAL_SET = [
    {"question": "What are calculated fields?", "expected_source": "Tableau.pdf"},
    {"question": "How do I group data in sql?", "expected_source": "SQL notes.pdf"},
    {"question": "What are nested query?", "expected_source": "SQL notes.pdf"},
    {"question": "How to aggregate data with pandas?", "expected_source": "Chap 3_Data manipulation with pandas.pdf"},
    {"question": "How do I use pip?", "expected_source": "Ai Engineering Chap 8_ Software Engineering Principles in Python.pdf"}
]

def run_eval():
    correct = 0
    for case in EVAL_SET:
        answer, hits = answer_question(case["question"])
        retrieved_sources= [h["source"] for h in hits]
        hit_found = case["expected_source"] in retrieved_sources
        correct += hit_found
        print(f"Q: {case["question"]}")
        print(f" Expected source in top-k: {hit_found}")
        print(f" Answer: {answer[:150]}...\n")

    print(f"Retieval accuracy: {correct}/{len(EVAL_SET)}")


if __name__ == "__main__":
    run_eval()