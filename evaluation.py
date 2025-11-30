import time
import csv
from datetime import datetime

from core.orchestrator import answer_question_with_rag, multi_agent_answer
from retrieval.vector_store import LocalVectorStore

INDEX_NAME = "edge_ai_paper"   # Or your active index
MODE = "multi"                # "simple" or "multi"
OUTPUT_FILE = "evaluation_results.csv"


QUESTIONS = [
    "Explain the pruning strategy in this paper",
    "What are the main contributions?",
    "What evaluation metrics are used?",
    "What is the role of the C2f module?",
    "How much speed improvement is achieved?",
    "What hardware is used for testing?",
    "What is the model size after pruning?",
    "What quantization technique is used?",
]


def run_eval():
    rows = []

    for q in QUESTIONS:
        start = time.time()

        if MODE == "multi":
            result = multi_agent_answer(q, index_name=INDEX_NAME)
            answer = result["final_answer"]
        else:
            answer = answer_question_with_rag(q, index_name=INDEX_NAME)

        total_time = round(time.time() - start, 2)

        rows.append([
            datetime.now().strftime("%Y-%m-%d"),
            INDEX_NAME,
            MODE,
            q,
            total_time,
            len(answer)
        ])

        print("DONE:", q, "→", total_time, "sec")

    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Index", "Mode", "Question", "Latency(sec)", "AnswerLength"])
        writer.writerows(rows)

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    run_eval()
