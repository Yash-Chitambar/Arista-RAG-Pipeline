import google.generativeai as genai
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from config import (
    GOOGLE_API_KEY,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL
)

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
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Create embedding function - must match the one used in ingestion
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        # Get collection with embedding function
        self.collection = self.chroma_client.get_collection(
            name="documents",
            embedding_function=self.embedding_function
        )

    def query(self, query_text: str, num_results: int = 3) -> str:
        """
        Query the RAG system and generate a response
        """
        try:
            # Query ChromaDB - it will compute the embedding automatically
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(num_results, self.collection.count())
            )
            
            # Extract context from results
            context = ""
            if results and 'documents' in results and results['documents']:
                for doc in results['documents'][0]:
                    context += doc + "\n\n"
            
            if not context:
                return "No relevant context found to answer the query."
            
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
            
            return response.text
        except Exception as e:
            print(f"Error during query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}" 