# 🏥 Medical Research RAG Pipeline

A production-grade Retrieval Augmented Generation (RAG) system that searches real medical research papers from PubMed and answers clinical questions with cited sources.

---

## 📌 Overview

This project fetches real research papers from the PubMed API and builds an advanced RAG pipeline on top of them. Instead of relying on an LLM's training data, the system retrieves actual published research to generate accurate, cited answers to medical questions.

---

## 🏗️ Architecture

```
PubMed API → Fetch Papers → Chunk & Embed → ChromaDB
                                                ↓
User Query → Multi Query Retriever → Hybrid Search (BM25 + Semantic)
                                                ↓
                                      Retrieved Chunks
                                                ↓
                                         Groq LLM
                                                ↓
                                    Cited Answer + Sources
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq (LLaMA 3.1 8B) |
| Embeddings | Cohere (embed-english-light-v3.0) |
| Vector Store | ChromaDB |
| Keyword Search | BM25 |
| Semantic Search | Cohere Embeddings + ChromaDB |
| Hybrid Search | LangChain EnsembleRetriever |
| Multi Query | LangChain MultiQueryRetriever |
| Evaluation | DeepEval + Gemini 2.5 Flash |
| Framework | LangChain |
| Data Source | PubMed API (35M+ real papers) |

---

## 📂 Project Structure

```
medical-research-rag/
│
├── 01_fetch_papers.py      # Fetch real papers from PubMed API
├── 02_ingest.py            # Chunk, embed and store in ChromaDB
├── pipeline.py             # Full Advanced RAG chain
├── evaluate.py             # DeepEval evaluation with Gemini judge
│
├── data/
│   └── papers.json         # Fetched PubMed papers (auto-generated)
│
├── requirements.txt        # All dependencies
├── .env.example            # Environment variables template
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/medical-research-rag.git
cd medical-research-rag
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

```bash
# Create .env file
cp .env.example .env
```

Add your API keys to `.env`:

```env
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
GOOGLE_API_KEY=your_google_api_key
```

### 5. Fetch Real PubMed Papers

```bash
python 01_fetch_papers.py
```

### 6. Index Papers into ChromaDB

```bash
python 02_ingest.py
```

### 7. Run the RAG Pipeline

```bash
python pipeline.py
```

### 8. Evaluate the Pipeline

```bash
python evaluate.py
```

---

## 🔑 API Keys (All Free)

| API | Get it here | Used for |
|---|---|---|
| Groq | https://console.groq.com | LLM for RAG answers |
| Cohere | https://cohere.com | Embeddings |
| Google Gemini | https://aistudio.google.com/apikey | Evaluation judge |

---

## 💡 Advanced RAG Techniques Used

| Technique | Purpose |
|---|---|
| **Hybrid Search** | Combines BM25 keyword + semantic search for better recall |
| **Multi Query Retrieval** | Generates multiple phrasings to find more relevant chunks |
| **Parent Document Retrieval** | Searches small chunks but returns full document context |
| **Contextual Compression** | Extracts only relevant sentences from retrieved chunks |

---

## 📊 Evaluation Results

Evaluated using DeepEval with Gemini 2.5 Flash as judge:

| Metric | Score | Status |
|---|---|---|
| Faithfulness | 0.88 | ✅ Excellent |
| Answer Relevancy | 1.00 | ✅ Perfect |

> Score guide: 0.8+ excellent · 0.6–0.8 good · below 0.5 needs work

---

## 🧪 Sample Questions

```
"What are the most effective treatments for Type 2 diabetes?"
"What are the side effects of metformin in elderly patients?"
"How does COVID-19 affect the cardiovascular system?"
"What is the relationship between sleep and mental health?"
```

---

## 📈 Future Improvements

- [ ] Add Streamlit web interface
- [ ] Support PDF upload for custom documents
- [ ] Add more evaluation metrics (Context Precision, Context Recall)
- [ ] Schedule automatic paper fetching for new research
- [ ] Add support for multiple languages

---

## 👨‍💻 Author

**Athar Abbas**  
AI Engineering Student

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).