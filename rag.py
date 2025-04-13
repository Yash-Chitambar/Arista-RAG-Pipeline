import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
from llama_index.core.settings import Settings
from ingestion import GoogleGenAIEmbedding
from config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    PINECONE_API_KEY
)

#Cutoff for what amount of relevance grade is allowed
RELEVANCE_THRESHOLD = 5

# Initialize Pinecone with new API
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "quickstart"

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
        
        # Create custom Google embedding model
        self.embed_model = GoogleGenAIEmbedding(model_name=EMBEDDING_MODEL, api_key=GOOGLE_API_KEY)
        
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

    def evaluate_relevance(self, context: str, query_text: str) -> int:
        grading_prompt = f"""
 You are evaluating the relevance of retrieved context based on a given question.
 Please grade the relevance of the provided context to the given question on a scale from 0 to 10,
 where 0 means "completely irrelevant" and 10 means "perfectly relevant"
 
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
            print(f"Error during relevance check: {str(e)}")
            return 0 # Defaults to 0 if error occurs

    def query(self, query_text: str, num_results: int = 3) -> str:
        """
        Query Pinecone directly for relevant documents and use Google Gemini for response generation
        """
        try:
            # Retrieve documents from Pinecone using the retriever
            nodes = self.retriever.retrieve(query_text)
            
            if not nodes:
                return "No relevant content found to answer the query."
                
            # Evaluate relevance of each retrieved node with Gemini
            relevant_context = []
            for node in nodes:
                relevance_score = self.evaluate_relevance(node.text, query_text)
                if relevance_score >= RELEVANCE_THRESHOLD:
                    relevant_context.append(node.text)
            
            if not relevant_context:
                return "No highly relevant context found to answer the query."
            
            # Join relevant context with double newlines
            context = "\n\n".join(relevant_context)
            
            # Prepare prompt for Gemini
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
            
            return response.text
        except Exception as e:
            print(f"Error during query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}" 