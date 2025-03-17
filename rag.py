import google.generativeai as genai
import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np
from config import (
    GOOGLE_API_KEY,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL
)

#Cutoff for what amount of relevance grade is allowed
RELEVANCE_THRESHOLD = 5


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
        
        # Create Google embedding function
        self.embedding_function = GoogleEmbeddingFunction(model_name=EMBEDDING_MODEL)
        
        # Get collection with embedding function
        self.collection = self.chroma_client.get_collection(
            name="documents",
            embedding_function=self.embedding_function
        )
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

    def query(self, query_text: str, num_results: int = 3) -> str:
        """
        Query the RAG system and generate a response
        """
        try:
            # Query ChromaDB - it will compute the embedding automatically
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(num_results * 2, self.collection.count())  # Get more results to filter
            )
            
            # Extract context from results and evaluate relevance
            relevant_context = []
            if results and 'documents' in results and results['documents']:
                for doc in results['documents'][0]:
                    # Evaluate relevance of each document
                    relevance_score = self.evaluate_relevance(doc, query_text)
                    if relevance_score >= RELEVANCE_THRESHOLD:
                        relevant_context.append(doc)
            
            if not relevant_context:
                return "No relevant context found to answer the query."
            
            # Join relevant context with double newlines
            context = "\n\n".join(relevant_context)
            
            # Prepare prompt with filtered context
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