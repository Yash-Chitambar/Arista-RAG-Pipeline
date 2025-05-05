from dotenv import load_dotenv
import google.generativeai as genai
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
load_dotenv()
from llama_parse import LlamaParse
from llama_cloud_services import LlamaParse
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.schema import TextNode
from llama_index.core import Settings
from typing import List, Set, Dict, Any, Optional
import json
import chromadb
from chromadb.config import Settings as ChromaSettings
import datetime
import uuid
import os
from pathlib import Path
import re
from copy import deepcopy
from pinecone import Pinecone, ServerlessSpec
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from pydantic import Field, ConfigDict
from config import (
    GOOGLE_API_KEY,
    LLAMA_CLOUD_API_KEY,
    PINECONE_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL
)

# Initialize Pinecone with new API
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "quickstart"
DIMENSION = 768

# Make sure the embedding model is set to Google's model
print(f"Using embedding model: {EMBEDDING_MODEL}")
if not EMBEDDING_MODEL.startswith("models/"):
    print(f"Warning: EMBEDDING_MODEL value '{EMBEDDING_MODEL}' may not be a valid Google model. Setting to default 'models/embedding-001'")
    EMBEDDING_MODEL = "models/embedding-001"

# Custom embedding class for Google Generative AI with proper Pydantic implementation
class GoogleGenAIEmbedding(BaseEmbedding):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model_name: str = Field(default=EMBEDDING_MODEL)
    api_key: str = Field(default=GOOGLE_API_KEY)
    max_chunk_size: int = Field(default=8000)  # Maximum characters per chunk
    
    def __init__(self, model_name=EMBEDDING_MODEL, api_key=GOOGLE_API_KEY, **kwargs):
        # Force model name to be a valid Google model
        if not model_name.startswith("models/"):
            print(f"Warning: Model name '{model_name}' is not a valid Google model. Using default 'models/embedding-001'")
            model_name = "models/embedding-001"
            
        genai.configure(api_key=api_key)
        super().__init__(model_name=model_name, **kwargs)
        print(f"Initialized GoogleGenAIEmbedding with model: {self.model_name}")

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks that fit within the model's context window."""
        if len(text) <= self.max_chunk_size:
            return [text]
            
        chunks = []
        current_chunk = ""
        
        # Split by sentences to maintain context
        sentences = text.split('. ')
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 2 <= self.max_chunk_size:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

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
        """Get text embedding with chunking for long texts."""
        try:
            # Split text into chunks if needed
            chunks = self._chunk_text(text)
            
            # Get embeddings for each chunk
            chunk_embeddings = []
            for chunk in chunks:
                try:
                    result = genai.embed_content(
                        model=self.model_name,
                        content=chunk,
                        task_type="retrieval_document"
                    )
                    chunk_embeddings.append(result["embedding"])
                except Exception as e:
                    print(f"Error generating embedding for chunk: {str(e)}")
                    continue
            
            if not chunk_embeddings:
                print("No valid embeddings generated for any chunks")
                return [0.0] * DIMENSION
                
            # Average the embeddings if we have multiple chunks
            if len(chunk_embeddings) > 1:
                return [sum(emb) / len(chunk_embeddings) for emb in zip(*chunk_embeddings)]
            return chunk_embeddings[0]
            
        except Exception as e:
            print(f"Error in text embedding process: {str(e)}")
            return [0.0] * DIMENSION

    def _is_valid_embedding(self, embedding: List[float]) -> bool:
        """Check if an embedding is valid (not all zeros)."""
        return not all(v == 0.0 for v in embedding)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding asynchronously."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding asynchronously."""
        return self._get_text_embedding(text)
        
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get multiple text embeddings at once."""
        embeddings = []
        for text in texts:
            embedding = self._get_text_embedding(text)
            if self._is_valid_embedding(embedding):
                embeddings.append(embedding)
            else:
                print(f"Warning: Invalid embedding generated for text of length {len(text)}")
        return embeddings

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get multiple text embeddings at once asynchronously."""
        return [await self._aget_text_embedding(text) for text in texts]

# Initialize LLM and embedding settings
# Create a safer default embedding model
default_embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001", 
    api_key=GOOGLE_API_KEY
)
Settings.embed_model = default_embed_model
Settings.llm = None  # No LLM to avoid OpenAI dependency

class DocumentIngestion:
    def __init__(self):
        """Initialize the document ingestion class with Pinecone"""
        try:
            # Initialize Google Generative AI
            genai.configure(api_key=GOOGLE_API_KEY)
            
            # Verify embedding model name is valid for Google
            embedding_model_name = EMBEDDING_MODEL
            if not embedding_model_name.startswith("models/"):
                print(f"Warning: Embedding model '{embedding_model_name}' is not a valid Google model. Using default 'models/embedding-001'")
                embedding_model_name = "models/embedding-001"
            
            # Create custom Google embedding model
            self.embed_model = GoogleGenAIEmbedding(model_name=embedding_model_name, api_key=GOOGLE_API_KEY)
            print(f"DocumentIngestion using embedding model: {self.embed_model.model_name}")
            
            # Initialize Pinecone index with new API
            if INDEX_NAME not in pc.list_indexes().names():
                print(f"Creating new Pinecone index: {INDEX_NAME}")
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
            print(f"Successfully connected to Pinecone index: {INDEX_NAME}")
            
            # Create Pinecone vector store for LlamaIndex
            self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
            print("Successfully created Pinecone vector store")
            
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
            
            # Initialize Gemini model for metadata generation
            self.gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            print("Successfully initialized Gemini model")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def get_processed_files(self) -> Set[str]:
        """Get set of already processed files from Pinecone metadata"""
        try:
            stats = self.pinecone_index.describe_index_stats()
            # Since Pinecone doesn't offer a direct way to query all metadata, we'll keep track of processed files separately
            # In a real application, you might want to use a separate database for this
            if hasattr(self, '_processed_files'):
                return self._processed_files
            return set()
        except Exception as e:
            print(f"Error getting processed files: {str(e)}")
            return set()

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
{text[:2000]}  # Limit text length to avoid token limits
"""
            title_response = self.gemini_model.generate_content(title_prompt)
            title = title_response.text.strip()
            # Limit title length to 100 characters
            enhanced_metadata["title"] = title[:100] if len(title) > 100 else title
            print(f"Generated title: {enhanced_metadata['title']}")
        except Exception as e:
            print(f"Error generating title: {str(e)}")
            enhanced_metadata["title"] = "Untitled Section"
        
        # Generate relevant questions
        try:
            questions_prompt = f"""
Based on the following text, generate 2-3 specific questions that someone might ask about this content.
Focus on the key information, technical details, or unique aspects mentioned in the text.
Return ONLY the questions as a numbered list without any additional text or explanations.
Keep each question under 100 characters.

TEXT:
{text[:2000]}  # Limit text length to avoid token limits
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
                        # Limit each question to 100 characters
                        questions.append(clean_line[:100].strip())
            
            # Limit to 3 questions max
            questions = questions[:3]
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

    def _check_metadata_size(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if metadata size is within Pinecone's limits (40KB).
        If not, trim the metadata to fit within the limit.
        """
        # Serialize metadata to JSON to check its size
        metadata_json = json.dumps(metadata)
        metadata_size = len(metadata_json.encode('utf-8'))
        
        # If metadata is within the limit, return it as is
        if metadata_size <= 40000:  # 40KB in bytes
            return metadata
            
        # Log warning about metadata size
        print(f"Warning: Metadata size ({metadata_size} bytes) exceeds Pinecone's limit. Trimming metadata...")
        
        # Create a copy of metadata for trimming
        trimmed_metadata = metadata.copy()
        
        # Trim metadata items until it's under the limit
        # Start with relevant_questions as it's likely the largest
        if "relevant_questions" in trimmed_metadata:
            del trimmed_metadata["relevant_questions"]
            
        # Check size again
        metadata_json = json.dumps(trimmed_metadata)
        metadata_size = len(metadata_json.encode('utf-8'))
        
        # If still too large, trim text content in other fields
        if metadata_size > 40000 and "title" in trimmed_metadata:
            trimmed_metadata["title"] = trimmed_metadata["title"][:50]
            
        # One more size check
        metadata_json = json.dumps(trimmed_metadata)
        metadata_size = len(metadata_json.encode('utf-8'))
        
        # If still too large, remove all optional fields and keep only essential ones
        if metadata_size > 40000:
            essential_metadata = {
                "file_name": metadata.get("file_name", "unknown"),
                "page_number": metadata.get("page_number", 0),
                "node_id": metadata.get("node_id", 0)
            }
            return essential_metadata
            
        return trimmed_metadata

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
                # Limit markdown content to 10,000 characters to avoid embedding issues
                md_content = page["md"]
                if len(md_content) > 10000:
                    print(f"Truncating page {page['page']} content from {len(md_content)} to 10000 characters")
                    md_content = md_content[:10000]
                
                # Create text node from markdown content
                text_node = TextNode(
                    text=md_content,
                    metadata={
                        "page_number": page["page"],
                        "type": "text",
                        "source": os.path.basename(pdf_path),
                        "file_name": os.path.basename(pdf_path),
                        "file_path": pdf_path,
                        "processed_date": str(datetime.datetime.now())
                    }
                )
                text_nodes.append(text_node)
                
        print(f"Extracted {len(text_nodes)} text nodes")
        return text_nodes

    def ingest_documents(self, directory_path: str) -> None:
        """Ingest only new documents from a directory and store them in Pinecone"""
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
                
                # Keep track of processed files
                if not hasattr(self, '_processed_files'):
                    self._processed_files = set()
                self._processed_files.add(pdf_file)
                
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
                continue
        
        if not all_text_nodes:
            print("No text nodes extracted from documents.")
            return
            
        print(f"\nProcessing {len(all_text_nodes)} total text nodes...")
        
        # Enhance node metadata and create index
        print("\nEnhancing metadata and generating embeddings...")
        
        # Process each node with enhanced metadata
        valid_nodes = []
        for i, node in enumerate(all_text_nodes):
            try:
                # Prepare base metadata
                base_metadata = {
                    "source": node.metadata.get("file_path", "unknown"),
                    "file_name": node.metadata.get("file_name", "unknown"),
                    "page_number": node.metadata.get("page_number", 0),
                    "node_id": i
                }
                
                # Generate enhanced metadata
                enhanced_metadata = self.generate_enhanced_metadata(node.text, base_metadata)
                
                # Check and trim metadata if needed
                safe_metadata = self._check_metadata_size(enhanced_metadata)
                
                # Update node metadata with safe metadata
                node.metadata = safe_metadata
                
                # Generate embedding for the node
                embedding = self.embed_model._get_text_embedding(node.text)
                
                # Only keep nodes with valid embeddings
                if self.embed_model._is_valid_embedding(embedding):
                    node.embedding = embedding
                    valid_nodes.append(node)
                else:
                    print(f"Skipping node {i} due to invalid embedding")
                
                # Print progress every 10 nodes
                if (i + 1) % 10 == 0 or i == len(all_text_nodes) - 1:
                    print(f"Progress: {i + 1}/{len(all_text_nodes)} nodes processed")
                    
            except Exception as e:
                print(f"Error processing node {i}: {str(e)}")
                continue
        
        if not valid_nodes:
            print("No valid nodes with embeddings generated.")
            return
            
        print(f"\nSuccessfully processed {len(valid_nodes)} valid nodes out of {len(all_text_nodes)} total nodes")
        
        # Create storage context with Pinecone vector store
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Configure global settings to use our embedding model and no LLM
        Settings.embed_model = self.embed_model
        Settings.llm = None  # No LLM to avoid OpenAI dependency
        
        # Create index from nodes with valid embeddings using batching
        print("\nCreating vector store index with batched processing...")
        
        # Use smaller batches to avoid overwhelming Pinecone
        BATCH_SIZE = 10
        total_batches = (len(valid_nodes) + BATCH_SIZE - 1) // BATCH_SIZE
        
        # Process in batches
        for i in range(0, len(valid_nodes), BATCH_SIZE):
            batch_nodes = valid_nodes[i:i+BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            
            print(f"Processing batch {batch_num}/{total_batches} with {len(batch_nodes)} nodes...")
            
            try:
                # Create a temporary index for this batch
                temp_index = VectorStoreIndex(
                    nodes=batch_nodes,
                    storage_context=storage_context
                )
                
                print(f"Successfully indexed batch {batch_num}")
                
                # Save reference to the latest index
                self.index = temp_index
                
            except Exception as e:
                print(f"Error indexing batch {batch_num}: {str(e)}")
                print(f"Skipping this batch and continuing...")
                continue
        
        # Save the count for later reference
        self._last_indexed_count = len(valid_nodes)
        
        print(f"Successfully indexed {len(valid_nodes)} nodes in Pinecone with enhanced metadata")

    def search_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search documents in Pinecone"""
        # Create a retriever to search the vector store
        retriever = self.index.as_retriever(similarity_top_k=n_results)
        
        # Retrieve nodes
        nodes = retriever.retrieve(query)
        
        # Format results
        results = []
        for node in nodes:
            results.append({
                "text": node.text,
                "metadata": node.metadata,
                "score": node.score if hasattr(node, "score") else None
            })
            
        return results

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
            return 0

# Example usage
if __name__ == "__main__":
    ingestion = DocumentIngestion()
    
    # Check Pinecone index status
    doc_count = ingestion.get_document_count()
    if doc_count > 0:
        print(f"Found existing Pinecone index with {doc_count} documents")
        processed_files = ingestion.get_processed_files()
        print(f"Already processed files: {processed_files}")
    
    # Process documents directory
    ingestion.ingest_documents("./documents")



        
