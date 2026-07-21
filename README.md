# Chat With Your Docs 📄

A Retrieval-Augmented Generation (RAG) app that answers questions about your own documents — with source citations, so every answer is traceable back to the exact chunk it came from.

Built to explore the core engineering behind modern LLM applications: chunking strategy, semantic retrieval, prompt construction, and evaluation — not just wrapping an API call.



---

## The problem

LLMs don't know about your private documents and will confidently make things up if asked about them. This app grounds every answer in your actual document set, and shows exactly which source it drew from — so you can verify the answer instead of just trusting it.

## How it works

```
   Documents (.pdf, .txt)
            │
            ▼
   ┌─────────────────┐
   │  Chunking        │  split into ~500-token overlapping chunks
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  Embedding       │  Pinecone integrated inference (multilingual-e5-large)
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  Vector store    │  Pinecone serverless index
   └────────┬────────┘
            ▼
  User question ──► Semantic search ──► top-k relevant chunks
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │  Claude API      │  generates answer
                                    │  (Anthropic)     │  + numbered citations
                                    └────────┬────────┘
                                              ▼
                                      Answer + sources
                                      shown in Streamlit UI
```

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Vector database | [Pinecone](https://www.pinecone.io/) (serverless) | Managed, integrated embedding + retrieval in one API, no self-hosted infra |
| Embeddings | Pinecone integrated inference (`multilingual-e5-large`) | No separate embeddings provider/key needed |
| Generation | [Claude API](https://console.anthropic.com) (Anthropic) | Strong instruction-following for citation-constrained answers |
| Chunking | `tiktoken` | Token-accurate chunking rather than naive character splitting |
| UI | [Streamlit](https://streamlit.io/) | Fast to build, good enough for a functional demo |
| PDF parsing | `pypdf` | Extracts text from PDF sources |

## Key design decisions

- **Chunk size (500 tokens, 50 token overlap):** balances retrieval precision (small enough to isolate relevant info) against context loss (overlap prevents key sentences being split across chunk boundaries).
- **Citations by design, not as an afterthought:** the system prompt requires the model to cite chunk numbers for every claim, and the UI surfaces the raw source chunks so a user can verify (or catch a wrong) answer instead of just trusting it.
- **Managed vector DB over local/file-based storage:** chosen to reflect a more production-realistic setup, and to avoid re-embedding documents on every app restart.

## Setup

```bash
git clone <your-repo-url>
cd rag-project
python3 -m venv venv
source venv/bin/activate

pip install pinecone anthropic streamlit pypdf tiktoken python-dotenv
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=your_anthropic_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

Add your source documents (`.txt` or `.pdf`) to a `docs/` folder, then run:

```bash
python ingest.py      # chunk, embed, and store documents in Pinecone
streamlit run app.py  # launch the chat UI
```

## Project structure

```
rag-project/
├── .env
├── ingest.py       # load, chunk, embed, and store documents
├── retrieve.py      # semantic search over stored chunks
├── generate.py      # builds context + calls Claude for the final answer
├── app.py           # Streamlit chat UI
├── eval.py           # small hand-built retrieval accuracy eval
└── docs/            # source documents
```

## Evaluation

A hand-built set of question/expected-source pairs (`eval.py`) checks whether retrieval surfaces the right document for a given question.

> _Example: "Achieved X/Y retrieval accuracy on a Z-question eval set. The misses were mostly cases where [chunk size / overlap / a specific edge case] caused the relevant sentence to be split across chunk boundaries, which I addressed by [fix]."_
>
> *(Fill this in with your actual results once you run `eval.py` — this is the part of the project that best demonstrates iteration and engineering judgment, so it's worth being specific here.)*

## Limitations & next steps

- No reranking step yet — a cross-encoder reranker after initial retrieval would likely improve precision on ambiguous queries.
- No conversation memory — each question is currently answered independently of prior turns.
- No streaming — responses currently return all at once rather than token-by-token.
- Retrieval quality hasn't been tested against edge cases like tables, code blocks, or multi-column PDFs.

## License

MIT (or your preference)
