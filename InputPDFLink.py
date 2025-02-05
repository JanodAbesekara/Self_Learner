from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import fitz  # PyMuPDF for extracting text from PDFs
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Initialize Vector Database
PERSIST_DIRECTORY = "UpdateVectorDB"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

# ✅ Create Upload Directory
UPLOAD_FOLDER = "pdf_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Define expected request format (List of PDFs)
class PDFList(BaseModel):
    files: list[dict]  # List of objects (id, name, link)

@app.post("/pdf-link")
async def process_pdfs(data: PDFList):
    """Download multiple PDFs from given links, extract text, convert to vectors, and store in vector DB."""
    results = []
    
    for file in data.files:
        try:
            pdf_link = file["link"]
            filename = file["name"]
            file_path = os.path.join(UPLOAD_FOLDER, filename)

            # ✅ Step 1: Download the PDF
            response = requests.get(pdf_link, stream=True)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                raise HTTPException(status_code=400, detail=f"Failed to download {filename}")

            # ✅ Step 2: Extract text from PDF
            extracted_text = extract_text_from_pdf(file_path)
            print(f"\n🔹 Extracted text from {filename}:\n", extracted_text[:500])  # ✅ Print first 500 characters

            # ✅ Step 3: Convert text to vector embeddings & store in ChromaDB
            store_text_in_vector_db(filename, extracted_text)

            results.append({"filename": filename, "text": extracted_text[:500]})  # ✅ Return preview of text

        except Exception as e:
            return {"error": str(e)}

    return {"message": "PDFs processed and stored in vector database successfully", "files": results}

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using PyMuPDF (fitz)."""
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def store_text_in_vector_db(filename, text):
    """Convert text into vector embeddings and store in ChromaDB."""
    text_chunks = text.split(". ")  # ✅ Split text into smaller chunks for better embedding
    vector_store.add_texts(texts=text_chunks, metadatas=[{"source": filename}] * len(text_chunks))

    print(f"✅ Stored {len(text_chunks)} text chunks from {filename} into ChromaDB.")

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)