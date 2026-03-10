from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# Load FLAN-T5 model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def ask_question(query):

    # Retrieve top documents
    docs = vectorstore.similarity_search(query, k=3)

    # Combine retrieved text
    context = "\n".join([doc.page_content[:400] for doc in docs])

    # Prompt engineering (important)
    prompt = f"""
You are an AI research assistant.

Using the context below, answer the question clearly in 2-3 sentences.

Context:
{context}

Question: {query}

Answer:
"""

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    # Decode answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer