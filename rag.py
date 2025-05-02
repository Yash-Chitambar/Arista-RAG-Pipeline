import google.generativeai as genai
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from pathlib import Path
from dotenv import load_dotenv
import os
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
load_dotenv()

# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-pro")  # Default to gemini-pro if not specified
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")  # Default to ./chroma_db
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # Default to all-MiniLM-L6-v2

# Validate required environment variables
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is required but not found in environment variables")

# Cutoff for what amount of relevance grade is allowed
RELEVANCE_THRESHOLD = 5

# Define consistent ChromaDB settings
CHROMA_SETTINGS = ChromaSettings(
    anonymized_telemetry=False,
    is_persistent=True,
    persist_directory=str(CHROMA_PERSIST_DIR)
)

#Creating Wrapper Class over Gemini to run DeepEval for Hallucination Testing on images
class GeminiWrapper(DeepEvalBaseLLM):
    def __init__(self, generative_model):
        self.model = generative_model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return ""

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "Google Generative AI"

class RAGSystem:
    def __init__(self):
        # Configure Google Gemini with the correct API key
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Initialize Gemini model with the latest supported version
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        # Use the model specified in config
        model_name = LLM_MODEL
        print(f"Using Gemini model: {model_name}")
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config
        )
        
        # Initialize ChromaDB client with consistent settings
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=CHROMA_SETTINGS
        )
        


        # Create embedding function - must match the one used in ingestion
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        

        # Define HNSW index parameters for better search
        hnsw_config = {
            "hnsw:space": "cosine", 
            "hnsw:construction_ef": 128,
            "hnsw:search_ef": 128,
            "hnsw:M": 16
        }
        
        # Get or create text collection with embedding function
        try:
            self.text_collection = self.chroma_client.get_or_create_collection(
                name="text_documents",
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function
            )
            
            print("Successfully connected to ChromaDB collection")
        except Exception as e:
            print(f"Error connecting to collection: {str(e)}")
            raise

    #Helper function to grade the relevance (1-10) of the context selected (Hallucination prevention)
    def evaluate_relevance(self, context: str, query_text: str) -> int:
        grading_prompt = f"""
 You are evalutating the relevance of retrieved context based on a given question.
 Please grade the relevance of the provided context to the given question on a scale from 0 to 10,
 where 0 means "completeley irrelevant" and 10 means "perfectly relevant"
 
 Context:
 {context}
 
 Question:
 {query_text}
 
 Provide only the numeric score as your response.
 """
        try:
            grading_response = self.model.generate_content(grading_prompt)
            graded_score = int(grading_response.text.strip())
            print(f"Relevance score given: {graded_score}/10")
            return graded_score
        except Exception as e:
            print(f"Error during hallucination check {str(e)}")
            return 0 # Defaults to 0 if error occurs

    def _get_sample_images(self, query_text: str, num_results: int = 3):
        """Fallback method to get sample images when ChromaDB query fails"""
        try:
            # Find image files in the extracted_images directory
            image_dir = Path("documents/extracted_images")
            image_files = list(image_dir.glob("*.jpg"))[:num_results]
            
            return [{
                "image_path": str(img_file),
                "page_number": i+1,
                "type": "image",
                "source": f"Page {i+1}"
            } for i, img_file in enumerate(image_files)]
        except Exception as e:
            print(f"Error getting sample images: {str(e)}")
            return []

    def query(self, query_text: str, num_results: int = 3) -> str:
        """
        Query the RAG system and generate a response
        """
        try:
            # Query text collection
            text_results = self.text_collection.query(
                query_texts=[query_text],
                n_results=min(num_results, self.text_collection.count()),
                include=["documents", "metadatas"]  # Include metadatas in results
            )
            
            context = self.retrieve_context(query_text, num_results=num_results)
            
            if not context:
                print("No context found in the database")
                return "No relevant information found to answer the query."
            
            # Prepare prompt with context
            prompt = f"""
Given the following context information, please answer the question.
If the answer cannot be determined from the context, say "I don't know based on the provided context."

Context:
{context}

Question: {query_text}

Answer:
"""
            # Generate response with Gemini
            response = self.model.generate_content(prompt)
            wrapped_model = GeminiWrapper(self.model)
            # Use wrapped_model to test for hallucinations in context (Score from 0 - 1: (#relevant statements/#Total statements)
            # Uses custom evaluation function if deepeval fails
            try:
                context_metric = ContextualRelevancyMetric(threshold=0.5, model=wrapped_model)
                context_test_case = LLMTestCase(
                    input = query_text,
                    actual_output = response,
                    retrieval_context = [context]
                )
                context_metric.measure(context_test_case)
                print(context_metric.score, context_metric.reason)
                if (context_metric.score < 0.5):
                    print("Context doesn't pass deepeval context relevancy test")
                    return "I'm not confident enough to answer this question based on retrieved information"
            except Exception as e:
                print(f"Error in contextual relevancy check using deepeval: {str(e)}")
                print("Switching to custom relevance evaluation")
                relevance_score = self.evaluate_relevance(context, query_text)
                if (relevance_score < RELEVANCE_THRESHOLD):
                    print("Context doesn't pass custom context relevancy test")
                    return "I'm not confident enough to answer this question based on retrieved information"

            #Check generated response with context generation using DeepEval (Provides score from 0 - 1 where higher scores represent hallucinations)
            output_metric = HallucinationMetric(threshold=0.5, model=wrapped_model)
            output_test_case = LLMTestCase(
                input = query_text,
                actual_output = response,
                context = [context]
            )
            output_metric.measure(output_test_case)
            print(output_metric.score, output_metric.reason)
            if (output_metric.score >= 0.5):
                print("Context doesn't pass deepeval context relevancy test")
                return "I'm not confident enough to answer this question based on retrieved information"
            print("Context passes all hallucination tests.")
            return response.text
        except Exception as e:
            print(f"Error during query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"
        

    #Function to retrieve the context based off of any given query      
    def retrieve_context(self, query_text: str, num_results: int = 3) -> str:
        try:
            # Query text collection
            text_results = self.text_collection.query(
                query_texts=[query_text],
                n_results=min(num_results, self.text_collection.count()),
                include=["documents", "metadatas"]  # Include metadatas in results
            )
            
            # Extract context from text results and track sources
            context = ""
            if text_results and text_results['documents'] and text_results['documents'][0]:
                print("\nSource Documents:")
                print("-" * 50)
                for i, (doc, metadata) in enumerate(zip(text_results['documents'][0], text_results['metadatas'][0])):
                    if doc is not None:
                        source = metadata.get('source', 'Unknown')
                        page = metadata.get('page_number', 'Unknown')
                        doc_len = len(doc)
                        print(f"{i+1}. Source: {source} (Page {page})")
                        print(f"   Length: {doc_len} characters")
                        print(f"   Preview: {doc[:150]}...")
                        print()
                        context += doc + "\n\n"
                print("-" * 50)
                return context.strip()
            
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return ""