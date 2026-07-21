#ingest.py
import os 
from pypdf import PdfReader
import tiktoken

DOCS_DIR = 'docs'
CHUNK_SIZE = 500 # tokens per chunk
CHUNK_OVERLAP = 5 # tokens shared between consecutive chunks

encoding = tiktoken.get_encoding("cl100k_base")


def load_documents(docs_dir):
    """Read all .txt and .pdf files in docs_di, return list of (filename, text)."""
    documents = []
    for filename in os.listdir(docs_dir):
        path = os.path.join(docs_dir, filename)
        if filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                documents.append((filename,f.read()))
        elif filename.endswith(".pdf"):
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            documents.append((filename, text))
    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlaping chunks measured in tokens."""
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        start += chunk_size - overlap
    return chunks


def build_chunks():
    """Return a list of dict:{id, text, source, chunk_index}."""
    documents = load_documents(DOCS_DIR)
    all_chunks =[]
    for filename, text in documents:
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{filename}-{i}",
                "text": chunk,
                "source": filename,
                "chunk_index": i,            
            })
    return all_chunks


import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = "rag-docs"
EMBED_MODEL = "multilingual-e5-large"
EMBED_DIM = 1024
BATCH_SIZE = 90 # Pinecone's interference API caps batch size around 96 texts

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

def get_index():
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
    return pc.Index(INDEX_NAME)

def embed_texts(texts, input_type="passage"):
    """imput_type is 'passage' when embedding docs, 'query' when embedding a search query."""
    result = pc.inference.embed(
        model=EMBED_MODEL,
        inputs=texts,
        parameters={"input_type": input_type, "truncate": "END"},
    )
    return [r["values"] for r in result]


def embed_and_store(chunks):
    index = get_index()
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vectors = embed_texts([c["text"] for c in batch], input_type="passage")
        upserts = [
            {
                "id": c["id"],
                "values": vec,
                "metadata": {"source": c["source"], "chunk_index": c["chunk_index"], "text": c["text"]},
            }
            for c, vec in zip(batch, vectors)
        ]
        index.upsert(vectors=upserts)
        print(f"Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")
    print(f"Stored {len(chunks)} chunksn in Pinecone index '{INDEX_NAME}'")

        
if __name__ == "__main__":
    chunks = build_chunks()
    print(f"Loaded {len(chunks)} chunks from {DOCS_DIR}/")
    embed_and_store(chunks)

if __name__ == "__main__":
    chunks = build_chunks()
    print(f"Loaded {len(chunks)} chunks from {DOCS_DIR}/")
    print("Example chunk:\n", chunks[0]["text"][:300] if chunks else "No documents found")