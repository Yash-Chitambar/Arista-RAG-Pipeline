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

#Cutoff for what amount of relevance grade is allowed
RELEVANCE_THRESHOLD = 5

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
        
        # Get text and image collections with embedding function
        self.text_collection = self.chroma_client.get_collection(
            name="text_documents",
            embedding_function=self.embedding_function
        )
        
        self.image_collection = self.chroma_client.get_collection(
            name="image_documents",
            embedding_function=self.embedding_function
        )

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
 
 
 

    def query(self, query_text: str, num_results: int = 3, include_images: bool = True) -> str:
        """
        Query the RAG system and generate a response
        """
        try:
            # Query text collection
            text_results = self.text_collection.query(
                query_texts=[query_text],
                n_results=min(num_results, self.text_collection.count())
            )
            
            # Query image collection if requested
            image_results = None
            if include_images:
                image_results = self.image_collection.query(
                    query_texts=[query_text],
                    n_results=min(num_results, self.image_collection.count())
                )
            
            # Extract context from text results
            context = ""
            if text_results and 'documents' in text_results and text_results['documents']:
                context += "Text Content:\n"
                for doc in text_results['documents'][0]:
                    context += doc + "\n\n"
            
            # Add image information to context if available
            if image_results and 'metadatas' in image_results and image_results['metadatas']:
                context += "\nRelevant Images:\n"
                for metadata in image_results['metadatas'][0]:
                    image_path = metadata.get('image_path', 'Unknown path')
                    page_number = metadata.get('page_number', 'Unknown page')
                    page_content = metadata.get('page_content', '').strip()[:200]  # Limit context length
                    
                    context += f"- Image: {image_path} (Page {page_number})\n"
                    if page_content:
                        context += f"  Context: {page_content}...\n\n"
            
            # print(f"\nRetrieved Context:\n{context if context else "No Context Retrieved"}") #Temporary print statement to viz retrieved context

            if not context:
                return "No relevant information found to answer the query."

            #Evaluate and check relevance score
            relevance_score = self.evaluate_relevance(context, query_text)
 
            if relevance_score < RELEVANCE_THRESHOLD:
                print("Context doesn't pass hallucination test") # Delete later; Here to understand explicitly when hallucination test is failed
                return "I'm not confident enough to answer this question based on retrieved information"
            
            # Prepare prompt with context
            prompt = f"""
Given the following context information, please answer the question.
If the answer cannot be determined from the context, say "I don't know based on the provided context."
If there are relevant images mentioned in the context, incorporate that information into your response.

Context:
{context}

Question: {query_text}

Answer:
"""
            
            print("Context passes hallucination test.") # Delete Later. Just verifies that the context passed hallucination test
            # Generate response with Gemini
            response = self.model.generate_content(prompt)
            
            return response.text
        except Exception as e:
            print(f"Error during query: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}" 