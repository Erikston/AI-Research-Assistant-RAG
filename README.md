# AI Research Assistant – Retrieval-Augmented Generation (RAG)

An AI-powered research assistant that answers questions from research papers using **Retrieval-Augmented Generation (RAG)**.  
The system processes PDF research papers, converts them into embeddings, stores them in a vector database, retrieves relevant content, and generates context-aware answers using a language model.

---

## 🚀 Features

- Upload and process research papers (PDF)
- Extract and split document text into smaller chunks
- Generate vector embeddings for semantic search
- Store embeddings using FAISS vector database
- Retrieve relevant document chunks for queries
- Generate answers using FLAN-T5 language model
- Interactive web interface built with Flask

---

## 🧠 RAG Pipeline Architecture

PDF Documents  
↓  
Text Extraction  
↓  
Text Chunking  
↓  
Embeddings (Sentence Transformers)  
↓  
FAISS Vector Database  
↓  
Retriever  
↓  
FLAN-T5 Language Model  
↓  
Generated Answer

---

## 🛠 Tech Stack

- Python  
- Flask  
- LangChain  
- FAISS (Vector Database)  
- HuggingFace Transformers  
- Sentence Transformers  
- HTML, CSS 

---

## 📂 Project Structure

```
AI-Research-Assistant-RAG
│
├── app.py
├── ingest.py
├── rag_pipeline.py
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── documents/
│   └── research_papers.pdf
│
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/Erikston/AI-Research-Assistant-RAG.git
cd AI-Research-Assistant-RAG
```

Create a virtual environment:

```
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Step 1: Process Research Papers

```
python ingest.py
```

### Step 2: Start the Web Application

```
python app.py
```

### Step 3: Open the Web Interface

Open your browser and go to:

```
http://127.0.0.1:5000
```

Ask questions about the uploaded research papers.

---

## 💡 Example Use Cases

- Academic research assistance  
- Question answering from research papers  
- Knowledge retrieval from large document collections  
- AI-powered document analysis  

---

## 📌 Future Improvements

- Support for multiple document formats  
- Chat-based user interface  
- Cloud deployment  
- Hybrid semantic + keyword search  

---

