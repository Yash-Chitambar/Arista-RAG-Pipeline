from dotenv import load_dotenv
import google.generativeai as genai
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
load_dotenv()
from llama_parse import LlamaParse
from llama_cloud_services import LlamaParse
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.schema import TextNode
from llama_index.core import Settings
from typing import List, Set
import json
import chromadb
from chromadb.config import Settings as ChromaSettings
import datetime
import uuid
import os
from pathlib import Path

# Initialize LLM and embedding settings
Settings.llm = Gemini()
Settings.embed_model = GeminiEmbedding()

# Define consistent ChromaDB settings
CHROMA_SETTINGS = ChromaSettings(
    anonymized_telemetry=False,
    is_persistent=True,
    persist_directory="./chroma_db"
)

class DocumentIngestion:
    def __init__(self):
        """Initialize the document ingestion class with ChromaDB client"""
        try:
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=CHROMA_SETTINGS
            )
            print("Successfully initialized ChromaDB client")
            
            # Create embedding function for ChromaDB
            self.embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            print("Successfully created embedding function")
            
            # Create or get collection with embedding function
            self.text_collection = self.chroma_client.get_or_create_collection(
                name="text_documents",
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function
            )
            print("Successfully initialized text collection")
            
            
            # Set up LlamaParse
            self.system_prompt = """
            For any graphs, try to create a 2D table of relevant values, along with a description of the graph.
            For any schematic diagrams, MAKE SURE to describe a list of all components and their connections to each other.
            Make sure that you always parse out the text with the correct reading order.
            """
            
            self.parser = LlamaParse(
                result_type="markdown",
                use_vendor_multimodal_model=True,
                vendor_multimodal_model_name="gemini-2.0-flash-001",
                invalidate_cache=True,
                system_prompt=self.system_prompt
            )
            print("Successfully initialized LlamaParse")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def get_processed_files(self) -> Set[str]:
        """Get set of already processed files from ChromaDB metadata"""
        try:
            all_metadata = self.text_collection.get()["metadatas"]
            processed_files = {meta["source"] for meta in all_metadata if meta and "source" in meta}
            return processed_files
        except Exception as e:
            print(f"Error getting processed files: {str(e)}")
            return set()

    def process_json_results(self, md_json_objs, pdf_path: str):
        """Process JSON results from LlamaParse and extract text"""
        text_nodes = []
        
        # Get the job ID from the first object
        job_id = md_json_objs[0].get("job_id", "")
        if not job_id:
            print("Warning: No job_id found in LlamaParse results")
        else:
            print(f"Using LlamaParse job_id: {job_id}")
            
        # Process each page for text nodes
        for obj in md_json_objs:
            for page in obj["pages"]:
                # Create text node from markdown content
                text_node = TextNode(
                    text=page["md"],
                    metadata={
                        "page_number": page["page"],
                        "type": "text",
                        "source": os.path.basename(pdf_path),
                        "processed_date": str(datetime.datetime.now())
                    }
                )
                text_nodes.append(text_node)
                
        print(f"Extracted {len(text_nodes)} text nodes")
        return text_nodes

    def ingest_documents(self, directory_path: str) -> None:
        """Ingest only new documents from a directory and store them in ChromaDB"""
        # Check if directory exists
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist.")
            return
            
        # Get all PDF files
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        if not pdf_files:
            print(f"No PDF files found in {directory_path}.")
            return

        # Get already processed files
        processed_files = self.get_processed_files()
        
        # Filter for new files
        new_files = [f for f in pdf_files if f not in processed_files]
        
        if not new_files:
            print("No new files to process. Skipping ingestion.")
            return
        
        print(f"Found {len(new_files)} new files to process: {new_files}")

        # Process each new PDF file
        all_text_nodes = []
        for pdf_file in new_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            print(f"\nProcessing {pdf_path}...")
            
            try:
                # Parse PDF and get JSON results
                json_objs = self.parser.get_json_result(pdf_path)
                
                # Process the JSON results
                text_nodes = self.process_json_results(json_objs, pdf_path)
                all_text_nodes.extend(text_nodes)
                
                print(f"Extracted {len(text_nodes)} text nodes from {pdf_path}")
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
                continue
        
        # Add text nodes to collection
        if all_text_nodes:
            print("\nAdding new text nodes to ChromaDB...")
            for i, node in enumerate(all_text_nodes):
                doc_id = str(uuid.uuid4())
                metadata = {
                    "source": node.metadata.get("source", "unknown"),
                    "page_number": node.metadata.get("page_number", 0),
                    "type": "text",
                    "node_id": i,
                    "processed_date": node.metadata.get("processed_date", str(datetime.datetime.now()))
                }
                
                self.text_collection.add(
                    ids=[doc_id],
                    documents=[node.text],
                    metadatas=[metadata]
                )
            
            print(f"Successfully added {len(all_text_nodes)} text nodes to ChromaDB")
        
        print(f"\nTotal documents in ChromaDB: {self.text_collection.count()}")

    def search_documents(self, query: str, n_results: int = 3) -> dict:
        """Search text documents"""
        results = self.text_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

    def get_document_count(self) -> int:
        """Get the number of documents in the collection"""
        return self.text_collection.count()

# Example usage
if __name__ == "__main__":
    ingestion = DocumentIngestion()
    
    # Check ChromaDB status
    doc_count = ingestion.get_document_count()
    if doc_count > 0:
        print(f"Found existing ChromaDB with {doc_count} documents")
        processed_files = ingestion.get_processed_files()
        print(f"Already processed files: {processed_files}")
    
    # Process documents directory
    ingestion.ingest_documents("./documents")
        



        
