import PyPDF2
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import Chroma  

# Load and extract text from the PDF
ExampleP = 'SE-Lesson1.pdf'
all_text = []

with open(ExampleP, 'rb') as pdffile:
    reader = PyPDF2.PdfReader(pdffile)
    
    # Extract text from each page and store in a list
    for pagenum in range(len(reader.pages)):
        page = reader.pages[pagenum]
        text = page.extract_text()
        all_text.append(text)  

# Embedding function to embed the data
def embed_data(data, embeddings):
    return embeddings.embed_documents(data)  

# Instantiate the HuggingFaceEmbeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Embed the extracted PDF text
embedded_texts = embed_data(all_text, embeddings)

# Initialize the Chroma vector store with the embeddings and original text
vector_store = Chroma.from_texts(texts=all_text, embedding=embeddings, persist_directory="AssignmentDB")

# Use the correct method to interact with the Chroma vector store
document_count = vector_store._collection.count()



print(f"Total number of documents stored: {document_count}")
print("Vector store initialized and stored in 'AssignmentDB' directory.")


