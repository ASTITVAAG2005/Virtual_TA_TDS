# Virtual_TA_TDS
# TDS Virtual TA: Retrieval-Augmented Assistant for Tools in Data Science 🤖📘

This repository presents a complete pipeline for building a Retrieval-Augmented Generation (RAG) based Virtual Teaching Assistant for the "Tools in Data Science" (TDS) course offered by IIT Madras. The assistant processes markdown lecture notes, forum posts, and image content to answer student queries effectively.

## 🧠 What is RAG?

Retrieval-Augmented Generation (RAG) combines vector search with language models. It first retrieves the most relevant context from a knowledge base and then feeds that to a generative model (like Gemini or GPT) to create grounded, context-aware responses.

---

## 🚀 Project Workflow Overview

### 🧰 Step 1: Clone the Course Repository

```bash
git clone https://github.com/sanand0/tools-in-data-science-public
```

Rename the folder appropriately (e.g., `Course_Content`) to reflect your project structure.

### 💻 Step 2: Set Up Python Environment

Create a virtual environment and install all required dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Packages used include: `fastapi`, `numpy`, `requests`, `google-generativeai`, `tiktoken`, `Pillow`, `beautifulsoup4`, `uvicorn`, `python-dotenv`.

---

### 📰 Step 3: Scrape Discourse Forum Posts

Use the custom script to collect threads from the official TDS Discourse forum:

* Domain: `https://discourse.onlinedegree.iitm.ac.in`
* Posts are saved individually in `Discourse_Content/`

**Authentication Required:**
Export your browser session cookies into `cookies.txt` using a browser extension to access protected forum posts.

---

### 🖼️ Step 4: Process and Describe Images

Discourse posts containing embedded images are downloaded into `downloaded_images/` using a helper script.

Descriptions for these images are generated using the Gemini 2.0 Flash model (`google.generativeai`) to enrich content. Rate limits (15 RPM) on the free tier are handled by batching requests and delaying appropriately.

---

### 🧾 Step 5: Construct `data.json`

Combine:

* Markdown lecture content
* Individual Discourse posts (not full threads)
* Image captions

The generated `data.json` has the following schema:

```json
{
  "title": "Post or Lecture Title",
  "source": "Full URL to the content",
  "filename": "Source file name",
  "content": "Combined text and image descriptions"
}
```

---

### 🧠 Step 6: Embedding with Compression

Content chunks are embedded using OpenAI's `text-embedding-3-small` via `aiproxy`. Metadata and embeddings are saved in compressed NumPy format (`data_embeddings.npz`).

**Why ********`.npz`******** Format?**

* Compact and fast to load
* Efficiently packs both vector and metadata
* Ideal for high-performance vector search systems

Chunking includes overlap to ensure context continuity. Empty content entries are automatically skipped.

---

### 🧪 Step 7: Build Queryable RAG API (`main.py`)

The backend script:

* Loads embeddings and metadata
* Accepts a text query and optional image
* Uses cosine similarity to find relevant chunks
* Calls Gemini 2.0 Flash to generate an answer

Start the local server using:

```bash
python main.py
```

---

### 🌐 Step 8: Sample API Call (Hosted on Vercel)

```bash
curl -X POST https://tds-project-new225-astitva-agarwals-projects.vercel.app/api/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG ? "}'
```

---

### ☁️ Step 9: Deploy to Vercel

```bash
npm install -g vercel
vercel login
vercel --prod
```

Set your API keys in the Vercel dashboard:

* `GEMINI_API_KEY`
* `AIPROXY_TOKEN`

(Found under Project Settings → Environment Variables)

---

## 🧠 Design Considerations & Challenges

* Extracted and processed over 900 individual posts for fine-grained context
* Handled image content gracefully with detailed descriptions
* Managed Gemini rate limits with batching and backoff
* Avoided hallucinations by grounding responses strictly in context

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
To add it manually:

```bash
echo "$(curl -s https://opensource.org/licenses/MIT | sed -n '/<pre>/,/<\/pre>/p' | sed 's/<[^>]*>//g')" > LICENSE
```

---

## 🙌 Acknowledgements

* Built for the Tools in Data Science course @ IIT Madras
* Combines the power of Gemini, OpenAI embeddings, and Discourse knowledge base

---

## 📬 Contact

**Astitva Agarwal**
📧 [Astitvaag2005@gmail.com](mailto:Astitvaag2005@gmail.com)
