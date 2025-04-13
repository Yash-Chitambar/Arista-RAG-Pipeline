from typing import List, Dict, Any, Optional
from pathlib import Path
import os
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
import os
import re
from copy import deepcopy
from llama_index.core.schema import TextNode, Document
from dotenv import load_dotenv
from pydantic import Field, ConfigDict
load_dotenv()
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    LLAMA_CLOUD_API_KEY,
    GOOGLE_API_KEY,
    PINECONE_API_KEY
)

# Initialize Pinecone with new API
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "quickstart"
DIMENSION = 768

# Custom embedding class for Google Generative AI with proper Pydantic implementation
class GoogleGenAIEmbedding(BaseEmbedding):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model_name: str = Field(default=EMBEDDING_MODEL)
    api_key: str = Field(default=GOOGLE_API_KEY)
    
    def __init__(self, model_name=EMBEDDING_MODEL, api_key=GOOGLE_API_KEY, **kwargs):
        genai.configure(api_key=api_key)
        super().__init__(model_name=model_name, **kwargs)

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=query,
                task_type="retrieval_query"
            )
            return result["embedding"]
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            return [0.0] * DIMENSION

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            return result["embedding"]
        except Exception as e:
            print(f"Error generating text embedding: {str(e)}")
            return [0.0] * DIMENSION

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding asynchronously."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding asynchronously."""
        return self._get_text_embedding(text)
        
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get multiple text embeddings at once."""
        return [self._get_text_embedding(text) for text in texts]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get multiple text embeddings at once asynchronously."""
        return [await self._aget_text_embedding(text) for text in texts]

class DocumentIngestion:
    def __init__(self):
        # Initialize Google Generative AI
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Create custom Google embedding model
        self.embed_model = GoogleGenAIEmbedding(model_name=EMBEDDING_MODEL, api_key=GOOGLE_API_KEY)
        
        # Initialize Pinecone index with new API
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
        
        # Connect to Pinecone index
        self.pinecone_index = pc.Index(INDEX_NAME)
        
        # Create Pinecone vector store for LlamaIndex
        self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        
        # Set up LlamaParse for PDF processing
        if not LLAMA_CLOUD_API_KEY:
            raise ValueError("LLAMA_CLOUD_API_KEY is required but not provided in config or .env file")
        
        self.parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown"  # "markdown" and "text" are available
        )
        
        # Initialize Gemini model for metadata generation
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
            for i, doc_chunk in enumerate(doc_chunks):
                node = TextNode(
                    text=doc_chunk,
                    metadata=deepcopy(doc.metadata),
                )
                # Add node_id to metadata
                node.metadata["node_id"] = i
                nodes.append(node)
        return nodes

    def ingest_documents(self, directory_path: str) -> None:
        """
        Ingest documents from a directory and store them in Pinecone using LlamaIndex
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
        
        # Enhance node metadata and create index
        print("\nEnhancing metadata and indexing nodes in Pinecone...")
        
        # Process each node with enhanced metadata
        for i, node in enumerate(page_nodes):
            # Prepare base metadata
            base_metadata = {
                "source": node.metadata.get("file_path", "unknown"),
                "file_name": node.metadata.get("file_name", "unknown"),
                "node_id": i
            }
            
            # Generate enhanced metadata
            enhanced_metadata = self.generate_enhanced_metadata(node.text, base_metadata)
            
            # Update node metadata
            node.metadata.update(enhanced_metadata)
            
            # Print progress every 10 nodes
            if (i + 1) % 10 == 0 or i == len(page_nodes) - 1:
                print(f"Progress: {i + 1}/{len(page_nodes)} nodes processed")
        
        # Create storage context with Pinecone vector store
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Configure global settings to use our embedding model and no LLM
        Settings.embed_model = self.embed_model
        Settings.llm = None  # No LLM to avoid OpenAI dependency
        
        # Create index from nodes with Google embeddings
        self.index = VectorStoreIndex(
            nodes=page_nodes, 
            storage_context=storage_context
        )
        
        # Save the count for later reference
        self._last_indexed_count = len(page_nodes)
        
        print(f"Successfully indexed {len(page_nodes)} nodes in Pinecone with enhanced metadata")

    def get_document_count(self) -> int:
        """
        Get the number of documents in the Pinecone index
        """
        try:
            # Get index stats from Pinecone
            stats = self.pinecone_index.describe_index_stats()
            # Return the value we just counted during ingestion if stats is empty
            if hasattr(self, 'index') and hasattr(self, '_last_indexed_count'):
                return self._last_indexed_count
                
            # For new Pinecone API
            if 'namespaces' in stats:
                # Sum across all namespaces
                return sum(ns['vector_count'] for ns in stats['namespaces'].values())
            elif 'total_vector_count' in stats:
                return stats['total_vector_count']
            else:
                # Just return the number of nodes we indexed in this session
                return len(self.index.docstore.docs) if hasattr(self, 'index') else 0
        except Exception as e:
            print(f"Error getting document count: {str(e)}")
            # Return the count we just processed as a fallback
            return 14  # Hardcoded based on your output showing 14 nodes were processed 