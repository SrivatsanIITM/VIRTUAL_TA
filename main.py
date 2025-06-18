
import subprocess

packages = [
    ["pip", "install", "requests", "openai", "tqdm"],
    ["pip", "install", "beautifulsoup4"],
    ["pip", "install", "fastapi[all]", "uvicorn", "langchain"],
    ["pip", "install", "--upgrade", "langchain", "langchain-community"],
    ["pip", "install", "faiss-cpu"],
    ["pip", "install", "sentence-transformers"],
    ["pip", "install", "tiktoken"],
    ["pip", "install", "langchain-huggingface"]
]

for cmd in packages:
    try:
        subprocess.run(cmd, check=True)
        print(f" Successfully ran: {' '.join(cmd)}")
    except subprocess.CalledProcessError as e:
        print(f" Failed to run: {' '.join(cmd)}")
        print(e)

import os
import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from typing import List, Dict

T_COOKIE = "PFJK13O%2FmR6aPveV07%2BiQaNC5tAEcA%2BnXu967VWQ6lAWMy%2FJyKD21oxvXjBW6NqCe4qKbtDgaASytchPBRbnbDYmqDjSW0B767D6sihoQT3dwD0jz46kVvMBlE30N%2BFHGszr5CQ3zX2Kar5Wc2rGlTGcIwhEoZSOAtYhnqh7TPKXZQPVJaV0XRMMJ9KeGorZsPJp7rzpuIxT4D%2F%2F60jQ7ZprdRsXdsaYrrX1%2BbitJMf1EjHWBf8yd60wNoMTqPmeQ%2FjeHSQfH0Nff9w%2F3NXd%2FiTbF35pPXdZEGueG%2FYgAcQzVFE7a8nuYCTuz4CLUXhP--qQ7WFdRbWSIsKKjs--M7wvxhuhUUYyeNKOKhUt3Q%3D%3D"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Cookie": f"_t={T_COOKIE}"
}

ROOT_URL = "https://discourse.onlinedegree.iitm.ac.in"

POST_URLS = [
    "/t/gpt-image-1-api-pricing/155987/2",
    "/t/ga5-question-8-clarification/155939/4",
    "/t/function-calling-clarity/155412",
    "/t/token-cost-of-gpt-4-turbo-and-mini/153778"
]

def extract_qa(post_url: str):
    full_url = ROOT_URL + post_url
    try:
        res = requests.get(full_url, headers=HEADERS)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        title_tag = soup.find("title")
        question = title_tag.text.replace(" - Discourse - Online Degree", "").strip() if title_tag else "Unknown Question"
        post_div = soup.find("div", class_="cooked")
        answer = post_div.get_text(separator="\n").strip() if post_div else question
        return {
            "question": question,
            "answer": answer[:500].strip(),
            "link": full_url
        }
    except Exception as e:
        print(f" Failed to fetch post {post_url}: {e}")
        return None
def main():
    qa_data = []
    for url in POST_URLS:
        print(f"🔍 Scraping: {url}")
        entry = extract_qa(url)
        if entry:
            qa_data.append(entry)
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/user_profiles_fallback.json", "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=2, ensure_ascii=False)
    print(" Saved Q&A data to data/raw/user_profiles_fallback.json")

if __name__ == "__main__":
    main()

from pathlib import Path

TDS_DIR = Path("tools-in-data-science-public")
documents = []
for md_file in TDS_DIR.glob("*.md"):
    text = md_file.read_text(encoding="utf-8")
    documents.append({
        "filename": md_file.name,
        "content": text
    })

print(f" Loaded {len(documents)} TDS modules")

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import base64
import io
from PIL import Image

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

os.environ["OPENAI_API_KEY"] = "your_openai_key_here"

with open("data/raw/user_profiles_fallback.json", "r", encoding="utf-8") as f:
    scraped_data = json.load(f)

for item in scraped_data:
    documents.append({"filename": item["link"], "content": item["answer"]})

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = []
for doc in documents:
    for i, chunk in enumerate(splitter.split_text(doc["content"])):
        chunks.append(Document(
            page_content=chunk,
            metadata={"source": doc["filename"], "chunk_id": i}
        ))

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
retriever = vectorstore.as_retriever()

def ask_query_simulated(question: str, image_base64: str = None):
    if image_base64:
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return {"error": f"Invalid image: {e}"}
    relevant_docs = retriever.get_relevant_documents(question)
    if not relevant_docs:
        return {"answer": "Sorry, no relevant content found."}
    answer_text = relevant_docs[0].page_content.strip()
    source_url = relevant_docs[0].metadata.get("source", "")
    return {
        "answer": answer_text[:700],
        "links": [{"url": source_url, "text": "View source"}]
    }

app = FastAPI()

class QueryInput(BaseModel):
    question: str
    image: Optional[str] = None

with open("data/raw/user_profiles_fallback.json", "r", encoding="utf-8") as f:
    discourse_qa = json.load(f)

def format_link(md_filename):
    base_url = "https://github.com/your-org/your-repo/blob/main/tools-in-data-science-public/"
    return f"{base_url}{md_filename}"

def get_discovered_docs(question: str, top_k: int = 2):
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    question_emb = model.encode(question, convert_to_tensor=True)
    scores_and_entries = []
    for entry in discourse_qa:
        ans_emb = model.encode(entry["answer"], convert_to_tensor=True)
        score = util.pytorch_cos_sim(question_emb, ans_emb).item()
        scores_and_entries.append((score, entry))
    top_entries = sorted(scores_and_entries, key=lambda x: x[0], reverse=True)[:top_k]
    return [{
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

print("Visit http://localhost:8000/docs in your browser")
# uvicorn.run(app, host="127.0.0.1", port=8000)
