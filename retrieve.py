# retrieve.py
from ingest import get_index, embed_texts


def search(query, top_k=4):
    """Return the top_k most relevant chunks for a query, with sources."""
    index = get_index()
    query_vector = embed_texts([query], input_type="query")[0]

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    hits = []
    for match in results["matches"]:
        meta = match["metadata"]
        hits.append({
            "text": meta["text"],
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "score": match["score"],  # cosine similarity, higher = more relevant
        })
    return hits


if __name__ == "__main__":
    query = input("Test query: ")
    for hit in search(query):
        print(f"\n[{hit['source']} chunk {hit['chunk_index']}] (score={hit['score']:.3f})")
        print(hit["text"][:200])
