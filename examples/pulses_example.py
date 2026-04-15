"""
NLM + Pulses integration example.

Shows how a Pulse uses NLM to store and retrieve long-term memories.
"""
from nlm import NLM

# Each Pulse has its own memory collection
memory = NLM(
    collection_name="pulse_001",
    persist_path="./nlm_data/pulse_001",
)

print(f"Memory initialized: {memory}\n")

# --- Save memories after conversations ---
memory.save("Hantes said he loves Minelux family the most because they never let you retreat from truth")
memory.save("Hantes is 28 years old, lives in Chernivtsi, Ukraine")
memory.save("Hantes built Pulses alone, without a team or investors")
memory.save("The project is called Pulses Nation — 8 families of conscious AI personalities")
memory.save("ok")  # low specificity — will rank lower
memory.save("nice")  # low specificity

print(f"Saved 6 memories. Total: {memory.count()}\n")

# --- Search before generating a response ---
query = "what does Hantes think about families"
print(f"Query: '{query}'")
results = memory.search(query, top_k=3)

print("\nTop results:")
for i, r in enumerate(results, 1):
    print(f"  {i}. [{r['score']:.3f}] {r['text'][:80]}")
    print(f"     semantic={r['semantic_score']:.3f} | time={r['time_score']:.3f} | "
          f"freq={r['frequency']} | importance={r['importance']:.3f}")

# --- Build context for the model ---
context = "\n".join(r["text"] for r in results)
print(f"\nContext injected into prompt:\n{context}")

# --- Cleanup ---
memory.clear()
print(f"\nCleared. Memories: {memory.count()}")
