import os
import json
import base64
import io
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ✅ Load scraped Q&A data
with open("data/raw/user_profiles_fallback.json", "r", encoding="utf-8") as f:
    scraped_data = json.load(f)

# ✅ Load markdown content
TDS_DIR = Path("tools-in-data-science-public")
documents = []
for md_file in TDS_DIR.glob("*.md"):
    text = md_file.read_text(encoding="utf-8")
    documents.append({
        "filename": md_file.name,
        "content": text
    })

# ✅ Add scraped content to documents
for item in scraped_data:
    documents.append({"filename": item["link"], "content": item["answer"]})

# ✅ Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = []
for doc in documents:
    for i, chunk in enumerate(splitter.split_text(doc["content"])):
        chunks.append(Document(
            page_content=chunk,
            metadata={"source": doc["filename"], "chunk_id": i}
        ))

# ✅ Vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
retriever = vectorstore.as_retriever()

# ✅ FastAPI app
app = FastAPI()

class QueryInput(BaseModel):
    question: str
    image: Optional[str] = None

def format_link(md_filename):
    base_url = "https://github.com/your-org/your-repo/blob/main/tools-in-data-science-public/"
    return f"{base_url}{md_filename}"

def get_discovered_docs(question: str, top_k: int = 2):
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    question_emb = model.encode(question, convert_to_tensor=True)
    scores_and_entries = []
    for entry in scraped_data:
        ans_emb = model.encode(entry["answer"], convert_to_tensor=True)
        score = util.pytorch_cos_sim(question_emb, ans_emb).item()
        scores_and_entries.append((score, entry))
    top_entries = sorted(scores_and_entries, key=lambda x: x[0], reverse=True)[:top_k]
    return [ {
        "answer": e["answer"][:300] + "...",
        "url": e["link"],
        "text": e["question"]
    } for _, e in top_entries]

@app.post("/ask")
async def ask_query(data: QueryInput):
    if data.image:
        try:
            image_data = base64.b64decode(data.image)
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    docs = retriever.get_relevant_documents(data.question)
    top_md = docs[:2]
    answers = [{
        "answer": doc.page_content[:300] + "...",
        "url": format_link(doc.metadata['source']),
        "text": f"From {doc.metadata['source']}"
    } for doc in top_md]

    discourse_answers = get_discovered_docs(data.question)
    combined_answers = answers + discourse_answers

    return JSONResponse(content={
        "question": data.question,
        "answers": combined_answers,
        "source_count": len(combined_answers)
    })

# ✅ Run server
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
