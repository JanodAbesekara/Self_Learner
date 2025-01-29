import requests
import io
import PyPDF2
from langchain.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_chroma import Chroma  # Updated import

# Function to download and read the PDF
def fetch_pdf_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        return io.BytesIO(response.content)  # Return as a file-like object
    except requests.exceptions.RequestException as e:
        print(f"Error fetching PDF: {e}")
        return None

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        all_text = [reader.pages[p].extract_text() for p in range(len(reader.pages))]
        return all_text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []

# Embedding function
def embed_data(data, embeddings):
    return embeddings.embed_documents(data)

# URLs of the PDFs
pdf_urls = [
    "https://firebasestorage.googleapis.com/v0/b/bytetcms.appspot.com/o/PDF%2F1720765368602_Handbook_of_Product_and_Service_Developm.pdf?alt=media&token=2d8f6a6c-c0bc-457e-a85e-abc4f8461f34", 
]

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize variables for storing all text and embeddings
all_texts = []

for url in pdf_urls:
    print(f"Processing PDF from URL: {url}")
    pdf_file = fetch_pdf_from_url(url)
    if pdf_file:
        extracted_text = extract_text_from_pdf(pdf_file)
        if extracted_text:
            all_texts.extend(extracted_text)
        else:
            print(f"No text extracted from: {url}")
    else:
        print(f"Failed to process URL: {url}")

if not all_texts:
    print("No text extracted from any PDF. Exiting...")
else:
    # Create the vector database with Chroma
    print("Embedding extracted text and creating vector database...")
    vector_store = Chroma.from_texts(texts=all_texts, embedding=embeddings, persist_directory="AnswerDB")

    # Check the count of stored documents
    document_count = vector_store._collection.count()
    print(f"Total number of documents stored: {document_count}")
    print("Vector store initialized and stored in 'AnswerDB' directory.")
