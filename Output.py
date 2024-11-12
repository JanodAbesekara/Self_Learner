import os
import sys
import signal
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Load API key for genai
Google_API_key = os.getenv("GEMINI_API")
genai.configure(api_key=Google_API_key)

# Signal handler for clean exit
def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class QueryResult(BaseModel):
    query: str

# Define prompt generation function
def generate_rag_prompt(query, context):
    escaped_context = context.replace("'", " ").replace('"', ' ').replace('\n', ' ')
    prompt = f"""
You are a knowledgeable academic assistant helping students understand course material. Based on the extracted information from the provided course material PDF, answer the following query with accuracy and reliability:

Question: {query}

Provide a clear, detailed answer with relevant information and concepts from the material. Aim to offer a complete explanation that will help the student gain a solid understanding of the topic.
    """
    return prompt

# Define a function to embed and store initial data for queries
def initialize_vector_db(texts):
    # Create the embeddings and vector store only if there is actual text content
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_texts(texts=texts, embedding=embeddings, persist_directory="./AssignmentDB")
    return vector_db

# Function to retrieve relevant context from the database
def get_relative_context_fromDB(query, vector_db):
    context = ""
    
    # Perform similarity search for relevant text
    search_results = vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    return context

# Generate an answer based on the prompt
def generate_answer(prompt):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# Main program loop
print("Enter your query: ")
query = input()

# Sample text content to initialize the vector DB
# Replace this with actual content extracted from a PDF or other sources
sample_texts = [
    "Introduction to programming and data structures.",
    "Advanced concepts in machine learning and artificial intelligence.",
    "Principles of software engineering and project management."
]

# Initialize the vector DB with sample content
vector_db = initialize_vector_db(sample_texts)

# Get context from the vector DB
context = get_relative_context_fromDB(query, vector_db)

# Generate a response
prompt = generate_rag_prompt(query, context)
response = generate_answer(prompt)
print(response)
