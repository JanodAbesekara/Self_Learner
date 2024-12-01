from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import PyPDF2
import os

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the vector database or connect if already exists
PERSIST_DIRECTORY = "NEwMODELDB"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

@app.get("/pdf-count/")
async def get_pdf_count(): 
    """
    Get the number of documents (PDFs or pages) stored in the vector database.
    """
    try:
        document_count = vector_store._collection.count()
        return {"pdf_count": document_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving PDF count: {str(e)}")

@app.get("/pdf-details/{pdf_index}/")
async def get_pdf_details(pdf_index: int):
    """
    Retrieve details (text) of a stored PDF by its index.
    """
    try:
        # Validate index
        if pdf_index < 0 or pdf_index >= vector_store._collection.count():
            raise HTTPException(status_code=404, detail="PDF index out of range")

        # Retrieve the document text (Chroma stores the original text as metadata)
        document = vector_store._collection.get(ids=[str(pdf_index)])
        document_text = document["documents"][0]  # Retrieve the document (text)

        return {"pdf_index": pdf_index, "pdf_text": document_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving PDF details: {str(e)}")

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, process it, and store the text embeddings in the vector database.
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Save the uploaded PDF locally
        temp_filename = "temp_uploaded.pdf"
        with open(temp_filename, "wb") as temp_file:
            temp_file.write(contents)

        # Extract text from the PDF
        all_text = []
        with open(temp_filename, "rb") as pdffile:
            reader = PyPDF2.PdfReader(pdffile)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():  # Only add non-empty text
                    all_text.append(text)

        # Clean up the temporary file
        os.remove(temp_filename)

        # Embed the extracted text and store it in the vector database
        if all_text:
            vector_store.add_texts(texts=all_text)
            vector_store.persist()
            return {
                "message": "PDF processed successfully.",
                "total_pages": len(all_text),
                "stored_vectors": len(all_text),
            }
        else:
            return {"message": "PDF processed, but no text was extracted."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

# Root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Welcome to the Vector Store API!"}
