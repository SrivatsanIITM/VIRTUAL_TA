import os
import json
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. Load markdown files
TDS_DIR = Path("tools-in-data-science-public")
documents = []
for md_file in TDS_DIR.glob("*.md"):
    text = md_file.read_text(encoding="utf-8")
    documents.append({"filename": md_file.name, "content": text})

# 2. Load scraped discourse data
with open("data/raw/user_profiles_fallback.json", "r", encoding="utf-8") as f:
    scraped_data = json.load(f)
for item in scraped_data:
    documents.append({"filename": item["link"], "content": item["answer"]})

# 3. Chunk the documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = []
for doc in documents:
    for i, chunk in enumerate(splitter.split_text(doc["content"])):
        chunks.append(Document(
            page_content=chunk,
            metadata={"source": doc["filename"], "chunk_id": i}
        ))

# 4. Create embeddings and FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)

# 5. Save FAISS index
vectorstore.save_local("db")
print("✅ FAISS index saved to 'db/' folder")
