from ingestion import DocumentIngestion
from rag import RAGSystem

def main():
    # Initialize the ingestion pipeline
    ingestion = DocumentIngestion()
    
    # Ingest documents from the documents directory
    ingestion.ingest_documents("documents")
    
    # Print the number of documents ingested
    print(f"Number of documents ingested: {ingestion.get_document_count()}")
    
    # Initialize the RAG system
    rag = RAGSystem()
    
    # Example query
    query = "What is Ultra Ethernet Consortium (UEC)?"
    response = rag.query(query)
    print(f"\nQuery: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main() 