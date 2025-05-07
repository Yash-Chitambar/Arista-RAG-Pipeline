import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
from llama_index.core.settings import Settings
from ingestion import GoogleGenAIEmbedding
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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")  # Default embedding model
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Validate required environment variables
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is required but not found in environment variables")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is required but not found in environment variables")

# Cutoff for what amount of relevance grade is allowed (Value from 0-1)
RELEVANCE_THRESHOLD = 0.33

# Initialize Pinecone with new API
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "quickstart"
DIMENSION = 768

#Creating Wrapper Class over Gemini to run DeepEval for Hallucination Testing
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
        
        # Verify embedding model is correct
        embedding_model_name = EMBEDDING_MODEL
        if not embedding_model_name.startswith("models/"):
            print(f"Warning: Embedding model '{embedding_model_name}' is not a valid Google model. Using default 'models/embedding-001'")
            embedding_model_name = "models/embedding-001"
            
        # Create custom Google embedding model
        self.embed_model = GoogleGenAIEmbedding(model_name=embedding_model_name, api_key=GOOGLE_API_KEY)
        print(f"RAGSystem using embedding model: {self.embed_model.model_name}")
        
        try:
            # Connect to Pinecone index
            self.pinecone_index = pc.Index(INDEX_NAME)
            
            # Create Pinecone vector store
            vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
            
            # Configure global settings to use our embedding model and no LLM
            Settings.embed_model = self.embed_model
            Settings.llm = None  # No LLM to avoid OpenAI dependency
            
            # Create a new index that uses the existing vectors in Pinecone
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store
            )
            
            # Create a simple retriever (not a query engine)
            self.retriever = self.index.as_retriever(
                similarity_top_k=10  # Retrieve documents for filtering
            )
            
            print(f"Successfully connected to Pinecone index: {INDEX_NAME}")
            
        except Exception as e:
            print(f"Error connecting to Pinecone: {str(e)}")
            raise ValueError(f"Failed to initialize Pinecone index: {str(e)}")

    #Helper function to grade the relevance (1-10) of the context selected (Hallucination prevention)
    def evaluate_relevance(self, context: str, query_text: str) -> int:
        grading_prompt = f"""
 You are evaluating the relevance of retrieved context based on a given question.
 Please grade the relevance of the provided context to the given question on a scale from 0 to 1 using decimal values,
 where 0 means "completely irrelevant" and 1 means "perfectly relevant"
 
 Context:
 {context}
 
 Question:
 {query_text}
 
 Provide only the numeric score as your response.
 """
        try:
            grading_response = self.model.generate_content(grading_prompt)
            graded_score = float(grading_response.text.strip())
            print(f"RELEVANCE SCORE GIVEN: {graded_score}")
            return graded_score
        except Exception as e:
            print(f"Error during relevance check: {str(e)}")
            return 0 # Defaults to 0 if error occurs

    def evaluate_relevance_deepeval(self, model, context: str, query_text: str) -> int:
        """
        Try to evaluate relevance using DeepEval. If it fails, fallback to Gemini scoring.
        Returns a decimal score from 0 to 1.
        """
        try:
            metric = ContextualRelevancyMetric(threshold=0.5, model=model)
            test_case = LLMTestCase(
                input=query_text,
                actual_output="",  # Not needed for relevance check
                retrieval_context=[context]
            )
            metric.measure(test_case)
            score = metric.score
            print(f"DeepEval relevance score: {score}")
            print(f"DeepEval relevance score reason: {metric.reason}")
            return score
        except Exception as e:
            print(f"DeepEval relevance evaluation failed: {str(e)}")
            print("Falling back to Gemini-based relevance scoring.")
            return self.evaluate_relevance(context, query_text)


    def _get_sample_images(self, query_text: str, num_results: int = 3):
        """Fallback method to get sample images when Pinecone query fails"""
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
        Query the RAG system and generate a response with hallucination detection
        """
        try:
            # Get the context from Pinecone
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

            wrapped_model = GeminiWrapper(self.model) #Model compatible with DeepEval that uses gemini

            #Check for contextual relevancy
            score = self.evaluate_relevance_deepeval(wrapped_model, context, query_text)
            if (score < RELEVANCE_THRESHOLD): 
                return "I'm not confident enough to answer this question based on retrieved information"

            #Check generated response with context generation using DeepEval (Provides score from 0 - 1 where higher scores represent hallucinations)
            output_metric = HallucinationMetric(threshold=0.5, model=wrapped_model)
            output_test_case = LLMTestCase(
                input = query_text,
                actual_output = response,
                context = [context]
            )
            output_metric.measure(output_test_case)
            print(f"Output hallucination score: {output_metric.score}")
            print(f"Output hallucination score reason: {output_metric.reason}")
            if (output_metric.score >= 0.5):
                print("Context doesn't pass deepeval hallucination test")
                return "I'm not confident enough to answer this question based on retrieved information"
                
            print("Context passes all hallucination tests.")
            return response.text
        except Exception as e:
            print(f"Error during query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"
        
    #Function to retrieve the context based off of any given query      
    def retrieve_context(self, query_text: str, num_results: int = 3) -> str:
        try:
            # Retrieve documents from Pinecone using the retriever
            nodes = self.retriever.retrieve(query_text)
            
            if not nodes:
                return ""
                
            # Evaluate relevance of each retrieved node with Gemini
            relevant_nodes = []
            print("\nSource Documents:")
            print("-" * 50)
            
            for i, node in enumerate(nodes):
                relevance_score = self.evaluate_relevance(node.text, query_text) #Uses custom evaluate_relevance as deepeval would be too slow
                if relevance_score >= RELEVANCE_THRESHOLD:
                    relevant_nodes.append(node)
                    source = node.metadata.get('file_name', 'Unknown')
                    page = node.metadata.get('page_number', 'Unknown')
                    doc_len = len(node.text)
                    print(f"{i+1}. Source: {source} (Page {page})")
                    print(f"   Relevance: {relevance_score}/10")
                    print(f"   Length: {doc_len} characters")
                    print(f"   Preview: {node.text[:150]}...")
                    print()
            
            print("-" * 50)
            
            if not relevant_nodes:
                return ""
            
            # Join relevant context with double newlines
            context = "\n\n".join([node.text for node in relevant_nodes])
            return context
            
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return ""