from ingestion import DocumentIngestion
from rag import RAGSystem
import os
import scrape

def main():
    print("\n========== SCRAPING ==========")

    scrape.scrape_and_download("www.arista.com/en/",max_files=4)


    # Initialize systems
    print("\n========== DOCUMENT INGESTION ==========")
    
    # Check if ChromaDB directory exists
    chroma_dir = "./chroma_db"
    if not os.path.exists(chroma_dir):
        print(f"No existing ChromaDB found. Will create new one at {chroma_dir}")
    else:
        print(f"Found existing ChromaDB at {chroma_dir}")
    
    ingestion = DocumentIngestion()
    doc_count = ingestion.get_document_count()
    print(f"Found {doc_count} documents")

    print("\n========== RAG SYSTEM ==========")
    rag = RAGSystem()
    
    # Check if ChromaDB exists and has documents
    if doc_count > 0:
        processed_files = ingestion.get_processed_files()
        print(f"Already processed files: {list(processed_files)}")
    else:
        print("No documents in ChromaDB yet. Please add PDFs to the documents directory.")
    
    # Check for new documents and process them
    documents_dir = "./documents"
    if not os.path.exists(documents_dir):
        print(f"\nCreating documents directory at {documents_dir}")
        os.makedirs(documents_dir)
        print("Please add PDF files to the documents directory and run the script again.")
        return
    
    # Check for PDF files
    pdf_files = [f for f in os.listdir(documents_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print("\nNo PDF files found in documents directory.")
        print("Please add PDF files to the documents directory and run the script again.")
        return
        
    print("\nChecking for new documents...")
    ingestion.ingest_documents(documents_dir)
    
    # Only proceed to query mode if we have documents
    if ingestion.get_document_count() == 0:
        print("\nNo documents in ChromaDB. Please add PDFs and run the script again.")
        return

    # Example query
    print("\n========== EXAMPLE QUERY ==========")
    query = "What is Arista Networks? What is their contact information?"
    print(f"Query: {query}")
    response = rag.query(query)
    print(f"\nResponse: {response}")

    # Start RAG query loop
    print("\n========== INTERACTIVE QUERY MODE ==========")
    print("Type 'exit' to quit")
    while True:
        query = input("\nEnter your question: ").strip()
        if query.lower() in ['exit', 'quit']:
            break
        
        if not query:
            continue
            
        try:
            response = rag.query(query)
            print("\nResponse:", response)
        except Exception as e:
            print(f"\nError processing query: {str(e)}")

if __name__ == "__main__":
    main() 