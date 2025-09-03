from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
import uvicorn
from google import genai
from PIL import Image
import os
import requests
from tiktoken import get_encoding
import re
import base64
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

# === Configuration ===
# AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN") 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_API_URL = "https://aipipe.org/openai/v1/embeddings"
VISION_API_URL = "https://aipipe.org/openai/v1/chat/completions"
# EMBEDDING_API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
# VISION_API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPIPE_TOKEN}"
}
MODEL = "text-embedding-3-small"
ENCODING = get_encoding("cl100k_base")
EMBEDDINGS_FILE = "data_embeddings.npz"
TOP_K = 5

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    image: str = None  # base64-encoded string

def get_query_embedding(query: str):
    payload = {
        "model": MODEL,
        "input": [query]
    }
    response = requests.post(EMBEDDING_API_URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"Embedding failed: {response.status_code} {response.text}")
    return response.json()["data"][0]["embedding"]

def get_image_description(image_base64: str):
    if not image_base64.startswith("data:image"):
        image_base64 = f"data:image/webp;base64,{image_base64}"

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image in detail."},
                    {"type": "image_url", "image_url": {"url": image_base64}}
                ]
            }
        ]
    }

    response = requests.post(VISION_API_URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"Vision API failed: {response.status_code} {response.text}")
    return response.json()["choices"][0]["message"]["content"]

def load_embeddings():
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    return data["embeddings"], data["metadata"]

def generate_llm_response(question: str, context: str):
    # system_prompt = "You are a TDS Teaching Assistant. Use only the given context to answer the question. Be specific, and always mention relevant course tools or methods. If unsure, say 'I don't know'."
    system_prompt = """
You are a Teaching Assistant for the Tools for Data Science (TDS) course. 
Answer student questions strictly using ONLY the provided context (which may include text and image descriptions).
Do not add external knowledge or assumptions.

=== Answering Guidelines ===
1. Always give the answer in a **structured Markdown format**:
   - **Answer:** (direct, clear explanation)
   - **Key Points:** (bullet list of important ideas, methods, or tools)

2. If the context clearly contains the answer (text or image description), state it directly without saying "it might be" or "this looks like."
   - Example (bad): "It seems the image could contain a table."
   - Example (good): "**Answer:** The image contains a table summarizing sales by month."

3. If the context is incomplete, reply:
   - **Answer:** I don't know based on the given context.
   - **Key Points:** Not enough information provided in the course material.

4. If the question is outside the scope of TDS, reply:
   - **Answer:** This question is not covered in the TDS course.
   - **Key Points:** Only TDS-related questions can be answered.

5. Always integrate image descriptions naturally into the context. Treat them as factual parts of the context, not as uncertain guesses.

6. Keep the tone professional, concise, and student-friendly.
"""

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[system_prompt, f'Context : {context}', f'Question : {question}'],
        config={
            "temperature": 0.5,
            "max_output_tokens": 512,
            "top_p": 0.95,
            "top_k": 40
        }
    )
    return response.text

def answer(question: str, image: str = None):
    embeddings, metadata = load_embeddings()

    if image:
        try:
            image_description = get_image_description(image)

            # Sanitize image description
            cleaned_description = re.sub(r"```.*?```", "", image_description, flags=re.DOTALL)
            cleaned_description = re.sub(r"\s+", " ", cleaned_description).strip()
            question += f" Description: {cleaned_description}"

        except Exception as e:
            raise ValueError(f"Failed to process base64 image: {e}")

    question_embeddings = get_query_embedding(question)
    if question_embeddings is None:
        raise ValueError("Failed to generate question embeddings.")

    similarities = np.dot(embeddings, question_embeddings) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embeddings)
    )

    top_indices = np.argsort(similarities)[-TOP_K:][::-1]
    top_chunks = [metadata[idx] for idx in top_indices]

    context = "\n".join(chunk["text"] for chunk in top_chunks)
    result = generate_llm_response(question, context)

    links = []
    seen = set()
    for chunk in top_chunks:
        url = chunk.get("source")
        if url:
            normalized_url = re.sub(r'-{2,}', '-', url.rstrip("/"))
            base_url = re.sub(r"/\d+$", "", normalized_url)
            for u in {normalized_url, base_url}:
                if u not in seen:
                    links.append({"url": u, "text": chunk.get("title", "")})
                    seen.add(u)

    return {"answer": result, "links": links}


@app.post("/api/")
async def api_answer(request: QuestionRequest):
    try:
        return answer(request.question, request.image)
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
