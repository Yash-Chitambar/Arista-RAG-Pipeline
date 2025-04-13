from ingestion import DocumentIngestion
from rag import RAGSystem
import os
import argparse
from config import LLAMA_CLOUD_API_KEY, PINECONE_API_KEY
from vdb import clear_index

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the RAG pipeline")
    parser.add_argument("--clear", action="store_true", help="Clear the index before ingestion")
    parser.add_argument("--ingest", action="store_true", help="Run the ingestion process")
    args = parser.parse_args()
    
    # Check if LlamaParse API key is available (only needed for ingestion)
    if args.ingest and not LLAMA_CLOUD_API_KEY:
        print("\nERROR: LLAMA_CLOUD_API_KEY is not set!")
        print("To fix this issue:")
        print("1. Create a .env file in the project root directory")
        print("2. Add the following line to the .env file:")
        print("   LLAMA_CLOUD_API_KEY=your-llama-parse-api-key-here")
        print("3. Get your API key from https://cloud.llamaindex.ai/")
        return
    
    # Check if Pinecone API key is available
    if not PINECONE_API_KEY:
        print("\nERROR: PINECONE_API_KEY is not set!")
        print("To fix this issue:")
        print("1. Add the following line to the .env file:")
        print("   PINECONE_API_KEY=your-pinecone-api-key-here")
        print("2. Get your API key from https://app.pinecone.io/")
        return

    # Check for PDF files only if ingestion is requested
    if args.ingest:
        # Create documents directory if it doesn't exist
        documents_dir = "documents"
        os.makedirs(documents_dir, exist_ok=True)
        
        # Check if there are any files in the documents directory
        files = os.listdir(documents_dir)
        pdf_files = [f for f in files if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"\nNo PDF files found in the {documents_dir} directory.")
            print("Please add PDF files to this directory before running the ingest process.")
            return
    
    try:
        # Clear the index if requested
        if args.clear:
            print("\n========== CLEARING INDEX ==========")
            clear_index()
        
        # Only run ingestion if requested
        if args.ingest:
            # Initialize the ingestion pipeline
            print("\n========== DOCUMENT INGESTION ==========")
            ingestion = DocumentIngestion()
            
            # Ingest documents from the documents directory
            ingestion.ingest_documents("documents")
            
            # Print the number of documents ingested
            doc_count = ingestion.get_document_count()
            print(f"\nNumber of document chunks ingested: {doc_count}")
            
            if doc_count == 0:
                print("No documents were ingested. Please check your PDF files and try again.")
                return
        
        # Initialize the RAG system
        print("\n========== RAG SYSTEM INITIALIZATION ==========")
        rag = RAGSystem()
        
        # Example query
        print("\n========== EXAMPLE QUERY ==========")
        query = "What are the key points about the topic?"
        print(f"Query: {query}")
        
        response = rag.query(query)
        print(f"\nResponse: {response}")
        
        # Interactive query loop
        print("\n========== INTERACTIVE QUERY MODE ==========")
        print("Type 'exit' to quit")
        
        while True:
            user_query = input("\nEnter your question: ")
            if user_query.lower() == 'exit':
                break
                
            response = rag.query(user_query)
            print(f"\nResponse: {response}")
    except ValueError as e:
        print(f"\nERROR: {str(e)}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main() 