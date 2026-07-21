# generate.py

import os
from dotenv import load_dotenv
import anthropic
from retrieve import search

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided
document excerpts. Each excrpt is numbered.

Rules:
- If the answer isn't in the excerpts, say so clearly - do not make things up.
- After each claim, cite the excerpt number(s) it came from, like[1] or [1,3].
- Be concise and direct.
"""


def build_context_block(hits):
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] (source: {hit['source']}, chunck {hit['chunk_index']})\n{hit['text']}")
    return "\n\n".join(lines)


def answer_question(question, top_k=4):
    hits = search(question, top_k=top_k)
    context = build_context_block(hits)

    user_message = f"""Document excerpts:
{context}

Question: {question}"""
    
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = "".join(block.text for block in response.content if block.type == "text")
    return answer_text, hits


if __name__ == "__main__":
    question = input("Ask a question about your docs: ")
    answer, hits = answer_question(question)
    print("\n--- ANSWER ---")
    print(answer)
    print("\n--- SOURCES ---")
    for i, hit in enumerate(hits, start=1):
        print(f"[{i}] {hit['source']} (chunk {hit['chunk_index']})")