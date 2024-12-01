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

# Define data model for query result
class QueryResult(BaseModel):
    query: str

# Define prompt generation function
def generate_rag_prompt(query, context):
    escaped_context = context.replace("'", " ").replace('"', " ").replace("\n", " ")
    prompt = f"""
Based on the extracted information, answer the following question directly and in a focused manner:

Question: {query}

Answer:
    """
    return prompt

# Define a function to embed and store initial data for queries
def initialize_vector_db(texts):
    if not texts:
        print("Error: No texts provided to initialize the vector database.")
        sys.exit(1)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma.from_texts(texts=texts, embedding=embeddings, persist_directory="./AssignmentDB")
        return vector_db
    except Exception as e:
        print(f"Error initializing vector database: {e}")
        sys.exit(1)

# Function to retrieve relevant context from the database
def get_relative_context_fromDB(query, vector_db):
    context = ""
    try:
        search_results = vector_db.similarity_search(query, k=6)
        for result in search_results:
            context += result.page_content + "\n"
    except Exception as e:
        print(f"Error retrieving context from database: {e}")
    return context

# Generate an answer based on the prompt
def generate_answer(prompt):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip() if response and response.text else "No response generated."
    except Exception as e:
        return f"Error generating answer: {e}"

# Main program loop
if __name__ == "__main__":
    # Replace sample_texts with actual content from a PDF or other sources
    sample_texts = [
        "Sample text 1 for initializing the vector database.",
        "Sample text 2 containing relevant information for course queries."
    ]

    # Initialize the vector DB with sample content
    vector_db = initialize_vector_db(sample_texts)

    while True:
        try:
            # Prompt the user for a query
            print("\nEnter your query (or type 'exit' to quit): ")
            query = input().strip()

            # Exit the loop if the user types "exit"
            if query.lower() == "exit":
                print("Goodbye!")
                break

            # Get context from the vector DB
            context = get_relative_context_fromDB(query, vector_db)

            # Generate a response
            prompt = generate_rag_prompt(query, context)
            response = generate_answer(prompt)
            print("\nResponse:")
            print(response)
        except Exception as e:
            print(f"An error occurred: {e}")
