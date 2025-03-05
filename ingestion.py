from typing import List
from pathlib import Path
import os
import chromadb
from chromadb.config import Settings as ChromaSettings
import uuid
from sentence_transformers import SentenceTransformer
from config import (
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)

class DocumentIngestion:
    def __init__(self):
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents"
        )
        
        # Initialize embedding model directly
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

    def ingest_documents(self, directory_path: str) -> None:
        """
        Ingest documents from a directory and store their embeddings
        """
        # Check if directory exists
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist.")
            return
            
        # Get all files in the directory
        file_paths = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.pdf') or file.endswith('.txt'):
                    file_paths.append(os.path.join(root, file))
        
        if not file_paths:
            print(f"No .pdf or .txt files found in {directory_path}.")
            return
            
        # Process each file
        for file_path in file_paths:
            try:
                # Simple text extraction (for txt files)
                if file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                # For PDFs, you'd need to use a PDF parser like PyPDF2 or pdfplumber
                else:
                    # Placeholder for PDF parsing
                    print(f"PDF processing for {file_path} would happen here")
                    text = f"Content from {file_path}"
                
                # Simple chunking by sentences
                sentences = text.replace('\n', ' ').split('.')
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip() + "."
                    if len(current_chunk) + len(sentence) <= CHUNK_SIZE:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Create embeddings and store chunks
                for i, chunk in enumerate(chunks):
                    doc_id = str(uuid.uuid4())
                    metadata = {
                        "source": file_path,
                        "chunk_id": i
                    }
                    
                    # Create embedding
                    embedding = self.embed_model.encode(chunk).tolist()
                    
                    # Add to collection
                    self.collection.add(
                        ids=[doc_id],
                        embeddings=[embedding],
                        metadatas=[metadata],
                        documents=[chunk]
                    )
                    
                print(f"Processed {file_path}: {len(chunks)} chunks created")
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

    def get_document_count(self) -> int:
        """
        Get the number of documents in the collection
        """
        return self.collection.count() 