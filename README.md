# Web Research Tool (RAG Based)

An AI-powered research assistant that extracts and summarizes insights from multiple news URLs using LLM + Retrieval-Augmented Generation (RAG).

## 🚀 Features
- Load multiple news URLs
- Convert content into vector embeddings (FAISS)
- Contextual compression for relevant information
- Ask natural language questions
- Gemini LLM powered answers
- Streamlit UI

## 🧠 Tech Stack
- LangChain
- Gemini (Google Generative AI)
- HuggingFace Embeddings
- FAISS
- Streamlit

## ⚙️ How it works
1. Load URLs
2. Create document chunks
3. Generate embeddings
4. Store in FAISS
5. Retrieve relevant context
6. LLM generates answer

## ▶️ Run locally

```bash
git clone https://github.com/dhvanityadav/Web-Research-Tool-RAG
cd Web-Research-Tool-RAG
pip install -r requirements.txt
streamlit run main.py
```
## 📌 Use Case
- News research
- Market analysis
- Competitive intelligence
- Academic research
- Deep Research

