from typing import List, Dict, Any
from pathlib import Path
import os
import chromadb
from chromadb.config import Settings as ChromaSettings
import uuid
import google.generativeai as genai
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
import os
import re  # Add this import at the top level
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
    LLAMA_CLOUD_API_KEY,
    GOOGLE_API_KEY
)

# Custom embedding function for ChromaDB that uses Google's embedding model
class GoogleEmbeddingFunction:
    def __init__(self, model_name):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model_name = model_name
        
    def __call__(self, input):
        """
        Generate embeddings for the given texts using Google's embedding model
        """
        if not input:
            return []
            
        embeddings = []
        for text in input:
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result["embedding"])
            except Exception as e:
                print(f"Error generating embedding: {str(e)}")
                # Return a zero vector as fallback
                embeddings.append([0.0] * 768)  # Typical embedding dimension
                
        return embeddings

class DocumentIngestion:
    def __init__(self):
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Create Google embedding function
        self.embedding_function = GoogleEmbeddingFunction(model_name=EMBEDDING_MODEL)
        
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
        
        # Initialize Gemini model for metadata generation
        genai.configure(api_key=GOOGLE_API_KEY)
        self.gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    def generate_enhanced_metadata(self, text: str, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate enhanced metadata including chapter title/summary and relevant questions
        using Google Gemini API.
        
        Args:
            text: The text content of the document chunk
            base_metadata: The base metadata dictionary
            
        Returns:
            Enhanced metadata dictionary with title and relevant questions
        """
        # Create a copy of the base metadata
        enhanced_metadata = base_metadata.copy()
        
        # Generate a title/summary for the chunk
        try:
            title_prompt = f"""
Generate a concise, descriptive title (5-10 words) for the following text that captures its main topic or theme.
Return ONLY the title without quotes, prefixes, or any additional text.

TEXT:
{text[:3000]}  # Limit text length to avoid token limits
"""
            title_response = self.gemini_model.generate_content(title_prompt)
            title = title_response.text.strip()
            enhanced_metadata["title"] = title
            print(f"Generated title: {title}")
        except Exception as e:
            print(f"Error generating title: {str(e)}")
            enhanced_metadata["title"] = "Untitled Section"
        
        # Generate relevant questions
        try:
            questions_prompt = f"""
Based on the following text, generate 3-5 specific questions that someone might ask about this content.
Focus on the key information, technical details, or unique aspects mentioned in the text.
Return ONLY the questions as a numbered list without any additional text or explanations.

TEXT:
{text[:4000]}  # Limit text length to avoid token limits
"""
            questions_response = self.gemini_model.generate_content(questions_prompt)
            questions_text = questions_response.text
            
            # Extract questions from response
            questions = []
            for line in questions_text.split('\n'):
                # Look for numbered or bullet point questions
                if line.strip():
                    # Remove numbers, bullets, etc.
                    clean_line = line.strip()
                    # Remove leading numbers, asterisks, dashes
                    clean_line = re.sub(r'^[\d\.\*\-]+\s*', '', clean_line)
                    if clean_line.endswith('?'):
                        questions.append(clean_line.strip())
            
            # Limit to 5 questions and join them into a single string with newlines
            questions = questions[:5]
            if questions:
                # Join questions into a single string with newline separators
                enhanced_metadata["relevant_questions"] = "\n".join(questions)
                print(f"Generated {len(questions)} relevant questions")
            else:
                enhanced_metadata["relevant_questions"] = ""
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            enhanced_metadata["relevant_questions"] = ""
        
        return enhanced_metadata

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
        print("\nAdding page nodes to ChromaDB with enhanced metadata...")
        
        for i, node in enumerate(page_nodes):
            # Create a unique ID for the document
            doc_id = str(uuid.uuid4())
            
            # Prepare base metadata
            base_metadata = {
                "source": node.metadata.get("file_path", "unknown"),
                "file_name": node.metadata.get("file_name", "unknown"),
                "node_id": i
            }
            
            # Generate enhanced metadata with title and relevant questions
            enhanced_metadata = self.generate_enhanced_metadata(node.text, base_metadata)
            
            # Add to collection - ChromaDB will compute embeddings automatically
            self.collection.add(
                ids=[doc_id],
                documents=[node.text],
                metadatas=[enhanced_metadata]
            )
            
            # Print progress every 10 nodes
            if (i + 1) % 10 == 0 or i == len(page_nodes) - 1:
                print(f"Progress: {i + 1}/{len(page_nodes)} nodes processed")
            
        print(f"Successfully added {len(page_nodes)} page nodes to ChromaDB with enhanced metadata")

    def get_document_count(self) -> int:
        """
        Get the number of documents in the collection
        """
        return self.collection.count() 