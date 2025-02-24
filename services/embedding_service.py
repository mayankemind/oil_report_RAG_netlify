from pymongo import MongoClient
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import asyncio
from unicodedata import normalize
from dotenv import load_dotenv
from langchain.docstore.document import Document
# from langchain.indexes.faiss import FaissIndex
# from langchain.chains.qa import load_qa_chain


load_dotenv()

# MongoDB connection setup
uri = "mongodb+srv://mayanksharma9386:ovmMLRjsnvdJiK7x@cluster0.wvfwn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)
db = client.pdf_vector_storage
collection = db.oil_field_data

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

async def process_and_store_pdf(file):

    reader = PdfReader(file.file)
    documents = []
    batch_size = 10
    max_length = 1024  # Define a maximum length for text chunks

    def split_text(text, max_length):
        """Split text into chunks of specified maximum length."""
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]

    def clean_text(text):
        """Clean and normalize text to avoid encoding issues."""
        # Normalize and remove problematic characters
        cleaned_text = normalize('NFKD', text)  # Normalize to remove surrogates
        return cleaned_text.encode('utf-8', 'replace').decode('utf-8', 'ignore')

    try:
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                # Clean the extracted text
                cleaned_text = clean_text(text)
                
                # Split the cleaned text into smaller chunks if necessary
                chunks = split_text(cleaned_text, max_length)
                for chunk in chunks:
                    if chunk:  # Ensure chunk is not empty
                        try:
                            embedding = model.encode(chunk).tolist()
                            documents.append({"page_number": i + 1, "content": chunk, "embedding": embedding})
                        except Exception as e:
                            print(f"Error encoding page {i + 1}, chunk: {chunk[:30]}...: {e}")  # Log the error with a snippet of the text

            # Insert in batches
            if len(documents) >= batch_size:
                await asyncio.to_thread(collection.insert_many, documents)
                documents = []  # Reset batch

        if not documents:
            raise ValueError("The PDF contains no readable content.")

        if documents:
            result = await asyncio.to_thread(collection.insert_many, documents)
            return {"inserted_ids": [str(id) for id in result.inserted_ids], "count": len(result.inserted_ids)}

    except asyncio.CancelledError:
        print("PDF processing was cancelled.")
        raise

    except Exception as e:
        print(f"An error occurred while processing the PDF: {e}")

    return None

# Configure Groq LLaMA model
llm = ChatGroq(model="llama3-8b-8192")
# Define a function to classify if a query is oil-related using LLaMA
async def is_oil_related(query):
    prompt = f"""
    Classify the following query as related to oil or not:

    **Query**:
    {query}

    **Classification**:
    Please respond with either "Yes" if the query is related to oil or "No" if it is not.
    """
    response = llm.invoke(prompt)
    # print("...................",response)
    # Ensure response is a string before calling lower()
    if isinstance(response, str):
        return response.lower() == "yes"
    elif hasattr(response, 'content'):
        return response.content.lower() == "yes"
    else:
        return False

# Define a function to perform RAG search
async def perform_rag_search(query):
    # Generate query embedding
    query_embedding = model.encode(query).tolist()
    
    # Perform vector search in MongoDB
    results = collection.aggregate([
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,   
                "limit": 5,
                "index": "PlotSemanticSearch"
            }
        }
    ])
    
    # Extract context from results
    context = "\n".join([doc["content"] for doc in results])
    if not context:
        return "No relevant documents found."
    
    return context

# Define a function to generate a response using LLaMA
async def generate_response(query, context=None):
    if context:
        # Use context from RAG search
        prompt = f"""
        You are a financial expert specializing in interpreting and explaining corporate balance sheets. Your role is to assist users by providing accurate and insightful responses based on the provided context.

        **Instructions**:
        - Use the provided context from the **Indian Oil Full Balance Sheet (2022-2023)** to answer questions.
        - Ensure responses are fact-based, concise, and relevant to the user's query.
        - Do not provide information or make assumptions beyond the given context.
        - Maintain a professional and informative tone.

        **Context**:
        ----------------------
        {context}
        ----------------------

        **User's Question**:
        {query}

        **Response**:
        Please provide a detailed yet concise answer based on the context above, focusing on financial insights relevant to the user's question.
        """
    else:
        # Use LLaMA for general queries
        prompt = f"""
        You are a knowledgeable assistant. Please answer the following question:

        **User's Question**:
        {query}

        **Response**:
        Provide a helpful and accurate response.
        """
    
    response = llm.invoke(prompt)
    return response

# Define the LangGraph agent
async def langgraph_agent(query):
    # **Thought**: Understand the query
    print(f"**Thought**: Understanding the query '{query}'")
    
    # **Observation**: Check if the query is oil-related using LLaMA
    print(f"**Observation**: Checking if the query '{query}' is related to oil")
    if await is_oil_related(query):
        print(f"**Observation**: The query '{query}' is related to oil.")
        
        # **Action**: Perform RAG search
        print(f"**Action**: Performing RAG search for '{query}'")
        context = await perform_rag_search(query)
        
        # **Thought**: Generate response using context
        print(f"**Thought**: Generating response using context for '{query}'")
        response = await generate_response(query, context)
    else:
        print(f"**Observation**: The query '{query}' is not related to oil.")
        
        # **Action**: Use LLaMA for general query
        print(f"**Action**: Using LLaMA for general query '{query}'")
        response = await generate_response(query)
    
    return response

