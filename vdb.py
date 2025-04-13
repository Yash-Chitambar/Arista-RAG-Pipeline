from config import PINECONE_API_KEY
from pinecone import Pinecone, ServerlessSpec

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

# To clear the index, uncomment and run: clear_index()

