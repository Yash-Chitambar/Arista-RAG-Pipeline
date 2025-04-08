from ingestion import DocumentIngestion
from rag import RAGSystem
import google.generativeai as genai
import os
from dotenv import load_dotenv
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-pro")

#Needs Wrapper becuase DeepEval is not directly compatible with gemini
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



def main():
    # Initialize systems
    print("\n========== DOCUMENT INGESTION ==========")
    
    # Check if ChromaDB directory exists
    chroma_dir = "./chroma_db"
    if not os.path.exists(chroma_dir):
        print(f"No existing ChromaDB found. Will create new one at {chroma_dir}")
    else:
        print(f"Found existing ChromaDB at {chroma_dir}")
    
    ingestion = DocumentIngestion()
    doc_count = ingestion.get_document_count()
    print(f"Found {doc_count} documents")

    print("\n========== RAG SYSTEM ==========")
    rag = RAGSystem()
    
    # Check if ChromaDB exists and has documents
    if doc_count > 0:
        processed_files = ingestion.get_processed_files()
        print(f"Already processed files: {list(processed_files)}")
    else:
        print("No documents in ChromaDB yet. Please add PDFs to the documents directory.")
    
    # Check for new documents and process them
    documents_dir = "./documents"
    if not os.path.exists(documents_dir):
        print(f"\nCreating documents directory at {documents_dir}")
        os.makedirs(documents_dir)
        print("Please add PDF files to the documents directory and run the script again.")
        return
    
    # Check for PDF files
    pdf_files = [f for f in os.listdir(documents_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print("\nNo PDF files found in documents directory.")
        print("Please add PDF files to the documents directory and run the script again.")
        return
        
    print("\nChecking for new documents...")
    ingestion.ingest_documents(documents_dir)
    
    # Only proceed to query mode if we have documents
    if ingestion.get_document_count() == 0:
        print("\nNo documents in ChromaDB. Please add PDFs and run the script again.")
        return
    
    # Generated Questions for the Images
    questions = []
    page_3_img_1_question = 'What are the two main components of the "Ethernet @ Scale: AI Center" network architecture?'
    questions.append(page_3_img_1_question)
    page_4_img_1_question_1 = "How much performance improvement does AI Optimized Load Balancing provide over Vanilla Ethernet at peak performance?"
    questions.append(page_4_img_1_question_1)
    page_4_img_1_question_2 = "How does AI Optimized Load Balancing compare to Vanilla Ethernet in terms of bandwidth performance across different message sizes?"
    questions.append(page_4_img_1_question_2)
    page_5_img_1_question_1 = "Which networking technology (Ethernet or InfiniBand) has a significantly lower relative failover delay and what is that value?"
    questions.append(page_5_img_1_question_1)
    page_5_img_1_question_2 = "How does the relative convergence time of InfiniBand compare to Ethernet in terms of failover delay?"
    questions.append(page_5_img_1_question_2)
    page_6_img_1_question_1 = "How does the relative bandwidth performance of Arista R4-AI compare to InfiniBand across different message sizes?"
    questions.append(page_6_img_1_question_1)
    page_6_img_1_question_2 = "Which networking technology (Arista R4-AI or InfiniBand) achieves higher NCCL All-to-All performance"
    questions.append(page_6_img_1_question_2)
    page_7_img_1_question_1 = "What is the primary advantage of using a Single Switch in terms of cost, power, and complexity?"
    questions.append(page_7_img_1_question_1)
    page_7_img_1_question_2 = "How many nodes can be supported by the Leaf-Spine and Distributed Etherlink Switch architectures?"
    questions.append(page_7_img_1_question_2)
    page_9_img_1_question = "How is the AI network structured in terms of AI Leaf, AI Host, and EOS connections?"
    questions.append(page_9_img_1_question)
    page_10_img_1_question = "What issue is causing the slowdown in the self-driving model training job 3014?"
    questions.append(page_10_img_1_question)
    page_11_img_1_question_1 = "How does AI XPU size scale with AI network options for different AI application sizes?"
    questions.append(page_11_img_1_question_1)
    page_11_img_1_question_2 = "What type of AI network option is associated with large-scale AI applications requiring 10K+ XPUs?"
    questions.append(page_11_img_1_question_2)
    page_11_img_2_question = "How are the XPU Accelerators connected within the AI platform architecture?"
    questions.append(page_11_img_2_question)
    page_12_img_2_question = "How does single-hop forwarding function within the AI platform's structure?"
    questions.append(page_12_img_2_question)

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-pro")
    google_model = genai.GenerativeModel(model_name=LLM_MODEL)

    wrapped_model = GeminiWrapper(google_model)

    # Use wrapped_model in HallucinationMetric
    metric = HallucinationMetric(threshold=0.5, model=wrapped_model)

    #Go through every question create an LLMTestCase to test model response on each question; Print Score and reasoning
    for question in questions:
        model_response = rag.query(question)
        model_context = [rag.retrieve_context(question)]

        test_case = LLMTestCase(
            input = question,
            actual_output = model_response,
            context = model_context
        )
        metric.measure(test_case)
        print(metric.score, metric.reason)
    





if __name__ == "__main__":
    main() 