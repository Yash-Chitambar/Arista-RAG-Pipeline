from typing import List
from pathlib import Path
import os
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings as ChromaSettings
import uuid
from sentence_transformers import SentenceTransformer
# bring in deps
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
import os
from copy import deepcopy
from llama_index.core.schema import TextNode
# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
load_dotenv()
from config import (
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    LLAMA_CLOUD_API_KEY
)

class DocumentIngestion:
    def __init__(self):
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Create embedding function for ChromaDB
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        # Create or get collection with embedding function
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_function
        )
        
        # Set up LlamaParse for PDF processing
        if not LLAMA_CLOUD_API_KEY:
            raise ValueError("LLAMA_CLOUD_API_KEY is required but not provided in config or .env file")
        
        self.parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown"  # "markdown" and "text" are available
        )

    def get_page_nodes(self, docs, separator="\n---\n"):
        """Split each document into page nodes by separator."""
        nodes = []
        for doc in docs:
            doc_chunks = doc.text.split(separator)
            for doc_chunk in doc_chunks:
                node = TextNode(
                    text=doc_chunk,
                    metadata=deepcopy(doc.metadata),
                )
                nodes.append(node)
        return nodes

    def ingest_documents(self, directory_path: str) -> None:
        """
        Ingest documents from a directory and store them in ChromaDB
        """
        # Check if directory exists
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist.")
            return
            
        # Get all PDF files from documents directory
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        print(f"Found PDF files: {pdf_files}")

        if not pdf_files:
            print(f"No PDF files found in {directory_path}.")
            return

        pdf_paths = set([os.path.join(directory_path, f) for f in pdf_files])
        print(f"PDF paths to process: {pdf_paths}")
            
        # Use SimpleDirectoryReader with LlamaParse to parse PDF files
        file_extractor = {".pdf": self.parser}
        documents = SimpleDirectoryReader(
            input_files=pdf_paths, 
            file_extractor=file_extractor
        ).load_data()
        
        print(f"\nProcessed {len(documents)} documents:")
        for doc in documents:
            print(f"- {doc.metadata.get('file_name', 'Unknown file')}")
            print(f"  Text length: {len(doc.text)}")
            print(f"  Metadata: {doc.metadata}")

        # Create page nodes from documents
        print("\nCreating page nodes...")
        page_nodes = self.get_page_nodes(documents)
        print(f"Created {len(page_nodes)} page nodes")

        # Write document text to .txt files (optional but useful for debugging)
        print("\nWriting document text to file...")
        for doc in documents:
            # Get the original PDF filename without extension
            pdf_filename = os.path.splitext(doc.metadata.get('file_name', 'document'))[0]
            # Create output filename
            output_filename = os.path.join(directory_path, f"{pdf_filename}.txt")
            
            # Write the text content to file
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(doc.text)
            
            print(f"Wrote text to: {output_filename}")
        
        # Process and add each page node to ChromaDB
        print("\nAdding page nodes to ChromaDB...")
        for i, node in enumerate(page_nodes):
            # Create a unique ID for the document
            doc_id = str(uuid.uuid4())
            
            # Prepare metadata
            metadata = {
                "source": node.metadata.get("file_path", "unknown"),
                "file_name": node.metadata.get("file_name", "unknown"),
                "node_id": i
            }
            
            # Add to collection - ChromaDB will compute embeddings automatically
            self.collection.add(
                ids=[doc_id],
                documents=[node.text],
                metadatas=[metadata]
            )
            
        print(f"Successfully added {len(page_nodes)} page nodes to ChromaDB")

    def get_document_count(self) -> int:
        """
        Get the number of documents in the collection
        """
        return self.collection.count() 