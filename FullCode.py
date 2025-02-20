import os
import sys
import signal
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests


app = FastAPI()

load_dotenv()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (Adjust if needed for security)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_DIRECTORY = "AnswerDB"
UPLOAD_FOLDER = "pdf_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Google API key for Gemini
Google_API_key = os.getenv("GEMINI_API")
if Google_API_key:
    genai.configure(api_key=Google_API_key)
else:
    print("Error: Google API key not found. Please set GEMINI_API in the environment.")
    sys.exit(1)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=DATABASE_DIRECTORY, embedding_function=embeddings)

# Define expected request format for PDF uploads
class FileSchema(BaseModel):
    id: str
    name: str
    link: str

class PDFList(BaseModel):
    files: list[FileSchema]  # Use a proper schema for files

# Define request body model for query API
class QueryRequest(BaseModel):
    query: str

# Signal handler for clean exit
def signal_handler(sig, frame):
    print("\nYou pressed Ctrl+C! Exiting program.")
    raise SystemExit(0)  # Corrected exit handling for FastAPI

signal.signal(signal.SIGINT, signal_handler)

# Function to generate the RAG (Retrieval-Augmented Generation) prompt
def generate_rag_prompt(query, context):
    escaped_context = context.replace("'", " ").replace('"', " ").replace("\n", " ")
    prompt = f"""
Based on the extracted information, answer the following question directly and in a focused manner:

Question: {query}

Answer:
    """
    return prompt

# Retrieve relevant context from the database
def get_relative_context_fromDB(query):
    context = ""
    try:
        search_results = vector_store.similarity_search(query, k=6)
        for result in search_results:
            context += result.page_content + "\n"
    except Exception as e:
        print(f"Error retrieving context from database: {e}")
    return context

# Generate an answer using Gemini AI
def generate_answer(prompt):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip() if response and response.text else "No response generated."
    except Exception as e:
        return f"Error generating answer: {e}"

@app.post("/ask")
async def ask_question(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Retrieve context from the database
    context = get_relative_context_fromDB(query)

    # Generate a response
    prompt = generate_rag_prompt(query, context)
    response = generate_answer(prompt)

    return {"query": query, "response": response}

@app.post("/pdf-link")
async def process_pdfs(data: PDFList):
    """Download multiple PDFs from given links, extract text, convert to vectors, and store in vector DB."""
    results = []
    
    for file in data.files:
        try:
            pdf_link = file.link
            filename = file.name
            file_path = os.path.join(UPLOAD_FOLDER, filename)

            # Step 1: Download the PDF
            response = requests.get(pdf_link, stream=True)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                raise HTTPException(status_code=400, detail=f"Failed to download {filename}")

            # Step 2: Extract text from PDF
            extracted_text = extract_text_from_pdf(file_path)
            print(f"\n🔹 Extracted text from {filename}:\n", extracted_text[:500])  # Print first 500 characters

            # Step 3: Convert text to vector embeddings & store in ChromaDB
            store_text_in_vector_db(filename, extracted_text)

            results.append({"filename": filename, "text": extracted_text[:500]})  # Return preview of text

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
    text_chunks = text.split(". ")  # Split text into smaller chunks for better embedding
    vector_store.add_texts(texts=text_chunks, metadatas=[{"source": filename}] * len(text_chunks))
    vector_store.persist()  # Ensure data is saved

    print(f"Stored {len(text_chunks)} text chunks from {filename} into ChromaDB.")

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)
