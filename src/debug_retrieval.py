# debug_retrieval.py

from query_rag import retrieve

QUESTION = (
    """
    Я встретила человека, поняла что ждала его всю жизнь, но он мало пишет и поздно отвечает. Как управлять своими эмоциями и жить без ожидания его внимания?
    """
)

TOP_K = 5

def main():
    docs = retrieve(QUESTION, top_k=TOP_K)

    print(f"\nQUESTION:\n{QUESTION}\n")
    print("=" * 80)

    for i, d in enumerate(docs, 1):
        print(f"\n[{i}] distance={d['distance']:.3f}")
        print(d["text"])
        print("-" * 80)

if __name__ == "__main__":
    main()
