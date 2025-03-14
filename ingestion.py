from typing import List
from pathlib import Path
import os
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings as ChromaSettings
import uuid
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
# bring in deps
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
import os
from copy import deepcopy
from llama_index.core.schema import TextNode, ImageNode
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
        
        # Create or get collections with embedding function
        self.text_collection = self.chroma_client.get_or_create_collection(
            name="text_documents",
            embedding_function=self.embedding_function
        )
        
        self.image_collection = self.chroma_client.get_or_create_collection(
            name="image_documents",
            embedding_function=self.embedding_function
        )
        
        # Set up LlamaParse for PDF processing
        if not LLAMA_CLOUD_API_KEY:
            raise ValueError("LLAMA_CLOUD_API_KEY is required but not provided in config or .env file")
        
        self.parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            use_vendor_multimodal_model=True,
            vendor_multimodal_model_name="anthropic-sonnet-3.5"
        )

    def extract_images_from_pdf(self, pdf_path: str, images_dir: str) -> List[dict]:
        """Extract images from PDF using PyMuPDF"""
        image_info = []
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Iterate through pages
        for page_num, page in enumerate(doc, 1):
            # Get images on the page
            image_list = page.get_images()
            
            # Process each image
            for img_idx, img in enumerate(image_list, 1):
                try:
                    # Get the image data
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    if base_image:
                        # Generate a unique filename
                        extension = base_image["ext"]
                        img_filename = f"page_{page_num}_img_{img_idx}.{extension}"
                        img_path = os.path.join(images_dir, img_filename)
                        
                        # Save the image
                        with open(img_path, "wb") as f:
                            f.write(base_image["image"])
                        
                        # Store image info
                        image_info.append({
                            "page": page_num,
                            "path": img_path,
                            "name": img_filename,
                            "type": "extracted_image",
                            "width": base_image.get("width"),
                            "height": base_image.get("height")
                        })
                        
                except Exception as e:
                    print(f"Error extracting image from page {page_num}: {str(e)}")
                    continue
        
        doc.close()
        print(f"Extracted {len(image_info)} images from PDF")
        return image_info

    def process_json_results(self, md_json_objs, base_path: str, pdf_path: str):
        """Process JSON results from LlamaParse and extract text/images"""
        text_nodes = []
        image_nodes = []
        
        # Create images directory if it doesn't exist
        images_dir = os.path.join(base_path, "extracted_images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Extract images using PyMuPDF
        image_dicts = self.extract_images_from_pdf(pdf_path, images_dir)
        print(f"Downloaded {len(image_dicts)} images to {images_dir}")
        
        # Create a mapping of page numbers to images
        page_to_images = {}
        for img in image_dicts:
            page_num = img["page"]
            if page_num not in page_to_images:
                page_to_images[page_num] = []
            page_to_images[page_num].append(img)
        
        # Process each page
        for obj in md_json_objs:
            for page in obj["pages"]:
                # Create text node from markdown content
                text_node = TextNode(
                    text=page["md"],
                    metadata={
                        "page_number": page["page"],
                        "type": "text",
                        "source": obj.get("source", "unknown")
                    }
                )
                text_nodes.append(text_node)
                
                # Process images for this page
                page_num = page["page"]
                if page_num in page_to_images:
                    for img in page_to_images[page_num]:
                        # Get surrounding text context
                        text_content = page["md"]
                        
                        # Create a searchable description
                        img_description = f"Image from page {page_num}"
                        if text_content:
                            # Add first paragraph of text content as context
                            paragraphs = [p for p in text_content.split("\n\n") if p.strip()]
                            if paragraphs:
                                img_description += f"\nContext: {paragraphs[0]}"
                        
                        img_node = ImageNode(
                            image_path=img["path"],
                            metadata={
                                "page_number": page_num,
                                "type": "image",
                                "image_type": img.get("type", "unknown"),
                                "source": obj.get("source", "unknown"),
                                "filename": img["name"],
                                "relative_path": os.path.join("extracted_images", img["name"]),
                                "width": img.get("width"),
                                "height": img.get("height"),
                                "page_content": text_content[:500] if text_content else ""
                            }
                        )
                        image_nodes.append((img_node, img_description))
        
        return text_nodes, image_nodes

    def clear_collections(self):
        """Clear all documents from both collections"""
        # Clear text collection
        text_ids = self.text_collection.get()["ids"]
        if text_ids:
            self.text_collection.delete(ids=text_ids)
            
        # Clear image collection
        image_ids = self.image_collection.get()["ids"]
        if image_ids:
            self.image_collection.delete(ids=image_ids)
            
        print("Cleared all documents from collections")

    def ingest_documents(self, directory_path: str) -> None:
        """
        Ingest documents from a directory and store them in ChromaDB
        """
        # Clear existing documents
        self.clear_collections()
        
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
        
        all_text_nodes = []
        all_image_nodes = []
        
        # Process each PDF file
        for pdf_path in pdf_paths:
            print(f"\nParsing {pdf_path}...")
            try:
                # Parse PDF and get JSON results
                md_json_objs = self.parser.get_json_result(pdf_path)
                
                # Process the JSON results
                text_nodes, image_nodes = self.process_json_results(md_json_objs, directory_path, pdf_path)
                all_text_nodes.extend(text_nodes)
                all_image_nodes.extend(image_nodes)
                
                print(f"Extracted {len(text_nodes)} text nodes and {len(image_nodes)} image nodes from {pdf_path}")
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
                continue

        # Add text nodes to text collection
        print("\nAdding text nodes to ChromaDB...")
        for i, node in enumerate(all_text_nodes):
            doc_id = str(uuid.uuid4())
            metadata = {
                "source": node.metadata.get("source", "unknown"),
                "page_number": node.metadata.get("page_number", 0),
                "type": "text",
                "node_id": i
            }
            
            self.text_collection.add(
                ids=[doc_id],
                documents=[node.text],
                metadatas=[metadata]
            )
        
        # Add image nodes to image collection
        print("\nAdding image nodes to ChromaDB...")
        for i, (node, description) in enumerate(all_image_nodes):
            doc_id = str(uuid.uuid4())
            metadata = {
                "source": node.metadata.get("source", "unknown"),
                "page_number": node.metadata.get("page_number", 0),
                "type": "image",
                "node_id": i,
                "image_path": node.metadata.get("relative_path", ""),
                "caption": node.metadata.get("caption", ""),
                "context_text": node.metadata.get("context_text", ""),
                "page_content": node.metadata.get("page_content", "")
            }
            
            self.image_collection.add(
                ids=[doc_id],
                documents=[description],  # Store the searchable description
                metadatas=[metadata]
            )
            
        print(f"Successfully added {len(all_text_nodes)} text nodes and {len(all_image_nodes)} image nodes to ChromaDB")

    def get_document_count(self) -> dict:
        """
        Get the number of documents in each collection
        """
        return {
            "text_documents": self.text_collection.count(),
            "image_documents": self.image_collection.count()
        }

    def search_images_by_description(self, description: str, n_results: int = 3) -> List[dict]:
        """
        Search for images by description and return a list of image metadata
        """
        # Query the image collection for images matching the description
        results = self.image_collection.query(
            query_texts=[description],
            n_results=n_results
        )

        # Extract image paths and metadata
        image_metadata = []
        for i, (doc_id, metadata) in enumerate(zip(results['ids'][0], results['metadatas'][0])):
            image_metadata.append({
                "doc_id": doc_id,
                "image_path": metadata['image_path'],
                "caption": metadata['caption'],
                "context_text": metadata['context_text'],
                "page_content": metadata['page_content'],
                "page_number": metadata['page_number']
            })

        return image_metadata 