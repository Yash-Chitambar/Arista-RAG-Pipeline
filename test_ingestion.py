from ingestion import DocumentIngestion

def main():
    # Initialize the DocumentIngestion class
    ingestion = DocumentIngestion()
    
    # Directory containing PDF files
    documents_dir = "documents"
    
    # Get initial document count
    initial_count = ingestion.get_document_count()
    print(f"Initial document count: {initial_count}")
    
    # Run ingestion
    print("\nStarting document ingestion...")
    ingestion.ingest_documents(documents_dir)
    
    # Get final document count
    final_count = ingestion.get_document_count()
    print(f"\nFinal document count: {final_count}")
    
    # Test image search functionality
    print("\nTesting image search...")
    image_results = ingestion.search_images_by_description("network diagram or architecture", n_results=3)
    
    print("\nFound images:")
    for img in image_results:
        print(f"\nImage path: {img['image_path']}")
        print(f"Caption: {img['caption']}")
        print(f"Context: {img['context_text']}")
        print(f"Page content: {img['page_content'][:200]}...")  # Show first 200 chars of page content

if __name__ == "__main__":
    main() 