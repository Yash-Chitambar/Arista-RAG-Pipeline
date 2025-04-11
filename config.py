import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = "AIzaSyBxTAoBoM9w0zVRCGLmEzpU6IXWyb6Ik8s"

# LlamaIndex API Key - this should be set in your .env file
# Example .env file entry: LLAMA_CLOUD_API_KEY=your-api-key-here
LLAMA_CLOUD_API_KEY = "llx-3euXrZkDJ6suHN0mCkr6mf8Fl3xAIjUyuaJUbUug6zPRz2YV"
if not LLAMA_CLOUD_API_KEY:
    print("Warning: LLAMA_CLOUD_API_KEY not found in environment. PDF parsing will fail.")
    print("Please create a .env file with your LLAMA_CLOUD_API_KEY.")

# ChromaDB settings
CHROMA_PERSIST_DIR = "./chroma_db"

# Document processing settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# LLM Model settings
LLM_MODEL = "gemini-1.5-flash"