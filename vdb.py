from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec



def clear_index():
    """Delete all vectors from the Pinecone index"""
    try:
        # Connect to the index
        index = pc.Index(index_name)
        
        # Delete all vectors (this is a more efficient operation than deleting vectors one by one)
        index.delete(delete_all=True)
        
        print(f"Successfully cleared all vectors from the '{index_name}' index")
    except Exception as e:
        print(f"Error clearing index: {str(e)}")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Get environment variables
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable is not set")

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create a dense index with integrated embedding
    index_name = "quickstart"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-2"
            )
        )
    index = pc.Index(index_name)
    print("Index stats:", index.describe_index_stats())

    # Uncomment to clear the index
    
    clear_index()
    print("--------cleared--------")
    print("Index stats:", index.describe_index_stats())

