import os
import json
import numpy as np
import requests
from tiktoken import get_encoding
from typing import List
from dotenv import load_dotenv

load_dotenv()
# ===== CONFIGURATION =====
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
DATA_FILE = "data.json"
OUTPUT_FILE = "data_embeddings.npz"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
EMBEDDING_API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}"
}
MODEL = "text-embedding-3-small"
ENCODING = get_encoding("cl100k_base")
# ==========================

def tokenize_text(text: str) -> List[int]:
    return ENCODING.encode(text)

def detokenize(tokens: List[int]) -> str:
    return ENCODING.decode(tokens)

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    tokens = tokenize_text(text)
    if len(tokens) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokens[start:end]
        chunks.append(detokenize(chunk))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    payload = {
        "model": MODEL,
        "input": texts
    }
    response = requests.post(EMBEDDING_API_URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"Embedding failed: {response.status_code} {response.text}")
    return [e["embedding"] for e in response.json()["data"]]

def process_and_embed(data_file: str):
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_chunks = []
    metadata = []
    skipped_chunks = []

    print("\n🧩 Chunking and preparing embeddings...")
    for idx, item in enumerate(data):
        title = item.get("title", "")
        source = item.get("source", "")
        filename = item.get("filename", "")
        content = item.get("content", "").strip()

        if not content:
            continue

        try:
            chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append({
                    "title": title,
                    "source": source,
                    "filename": filename,
                    "chunk_index": i,
                    "text": chunk
                })
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            skipped_chunks.append({"filename": filename, "error": str(e)})

    print(f"\n📊 Total chunks to embed: {len(all_chunks)}")

    embeddings = []
    valid_metadata = []
    batch_size = 10
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        meta_batch = metadata[i:i + batch_size]
        batch_num = i // batch_size + 1

        print(f"📦 Batch {batch_num} ({len(batch)} chunks)...")
        try:
            batch_embeddings = get_embeddings(batch)
            embeddings.extend(batch_embeddings)
            valid_metadata.extend(meta_batch)
        except Exception as e:
            print(f"❌ Error in batch {batch_num}: {e}")
            for m in meta_batch:
                skipped_chunks.append(m)

    print(f"\n✅ Finished embedding. Successful: {len(embeddings)} | Skipped: {len(skipped_chunks)}")

    print("💾 Saving to embeddings.npz...")
    np.savez(OUTPUT_FILE, embeddings=np.array(embeddings), metadata=np.array(valid_metadata, dtype=object))

    if skipped_chunks:
        with open("skipped_chunks.json", "w", encoding="utf-8") as f:
            json.dump(skipped_chunks, f, indent=2)
        print("🗂️ Skipped chunks written to skipped_chunks.json")

    print("\n🎉 Done!")

if __name__ == "__main__":
    process_and_embed(DATA_FILE)
