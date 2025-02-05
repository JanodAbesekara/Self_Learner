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

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (Adjust if needed for security)
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Load Google API key for Gemini
Google_API_key = os.getenv("GEMINI_API")
if Google_API_key:
    genai.configure(api_key=Google_API_key)
else:
    print("Error: Google API key not found. Please set GEMINI_API in the environment.")
    sys.exit(1)

# Signal handler for clean exit
def signal_handler(sig, frame):
    print("\nYou pressed Ctrl+C! Exiting program.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Define request body model
class QueryRequest(BaseModel):
    query: str

# Function to generate the RAG (Retrieval-Augmented Generation) prompt
def generate_rag_prompt(query, context):
    escaped_context = context.replace("'", " ").replace('"', " ").replace("\n", " ")
    prompt = f"""
Based on the extracted information, answer the following question directly and in a focused manner:

Question: {query}

Answer:
    """
    return prompt

# Initialize vector database
def initialize_vector_db(texts):
    if not texts:
        print("Error: No texts provided to initialize the vector database.")
        sys.exit(1)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma.from_texts(texts=texts, embedding=embeddings, persist_directory="./UpdateVectorDB")
        return vector_db
    except Exception as e:
        print(f"Error initializing vector database: {e}")
        sys.exit(1)

# Retrieve relevant context from the database
def get_relative_context_fromDB(query, vector_db):
    context = ""
    try:
        search_results = vector_db.similarity_search(query, k=6)
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

# Initialize the vector DB with sample content (Replace this with actual content)
sample_texts = [
    "Sample text 1 for initializing the vector database.",
    "Sample text 2 containing relevant information for course queries."
]
vector_db = initialize_vector_db(sample_texts)

# API Endpoint: Ask a Question
@app.post("/ask")
async def ask_question(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Retrieve context from the database
    context = get_relative_context_fromDB(query, vector_db)

    # Generate a response
    prompt = generate_rag_prompt(query, context)
    response = generate_answer(prompt)

    return {"query": query, "response": response}

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
