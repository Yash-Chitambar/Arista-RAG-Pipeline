import streamlit as st
from rag import RAGSystem
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Arista RAG Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# Sidebar content
with st.sidebar:
    st.title("Arista Networks RAG Chatbot")
    st.markdown("""
    This chatbot uses RAG (Retrieval Augmented Generation) to answer questions about Arista Networks.
    
    It retrieves information from Arista documentation stored in a Pinecone vector database.
    """)
    
    # Add information about the system
    st.subheader("System Information")
    st.info("""
    - Using Google Gemini for LLM generation
    - Using Pinecone for vector database
    - Using LlamaIndex for RAG pipeline
    """)
    
    # Add a clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialize RAG system (only once)
@st.cache_resource
def initialize_rag_system():
    try:
        rag = RAGSystem()
        return rag
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main title
st.title("Arista Networks Chatbot")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize RAG system
rag = initialize_rag_system()

if rag is None:
    st.error("Failed to initialize the RAG system. Please check your configuration.")
else:
    # Chat input
    if prompt := st.chat_input("Ask me about Arista Networks..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # Get response from RAG system
                with st.spinner("Retrieving information..."):
                    response = rag.query(prompt)
                
                # Update placeholder with full response
                message_placeholder.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Add some additional helpful UI elements
st.divider()
st.caption("This chatbot uses RAG to provide accurate information about Arista Networks based on their documentation.") 