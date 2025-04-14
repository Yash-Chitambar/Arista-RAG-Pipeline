# Arista RAG Pipeline

A comprehensive Retrieval-Augmented Generation (RAG) system that combines web scraping, document ingestion, and AI-powered question answering. The system is designed to work with Arista's website but can be configured for other domains.

## System Architecture

The system consists of three main components:

1. **Web Scraper** (`scrape.py`)
   - Performs BFS crawling of websites
   - Downloads documents (PDFs, docs, etc.)
   - Stores files in the `documents` directory

2. **Document Ingestion** (`ingestion.py`)
   - Processes downloaded documents
   - Uses LlamaParse for document parsing
   - Stores processed content in ChromaDB
   - Handles document metadata and embeddings

3. **RAG System** (`rag.py`)
   - Uses Google's Gemini model for question answering
   - Implements relevance scoring
   - Provides hallucination detection
   - Manages context retrieval and response generation

## Features

- **Web Scraping**
  - BFS-based web crawling
  - Document detection and downloading
  - Domain-specific crawling
  - File type filtering

- **Document Processing**
  - PDF parsing with LlamaParse
  - Text extraction and chunking
  - Metadata management
  - Vector embeddings

- **RAG Capabilities**
  - Context-aware question answering
  - Relevance scoring
  - Hallucination detection
  - Multi-document support

## Prerequisites

- Python 3.8+
- Poetry (Python package manager)
- Chrome browser and ChromeDriver
- Google API key for Gemini
- LlamaParse API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repo-name>
```

2. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Create and activate the Poetry environment:
```bash
poetry install --no-root
poetry env activate
```

4. Set up environment variables:
Create a `.env` file with:
```env
GOOGLE_API_KEY=your_google_api_key
LLAMA_PARSE_API_KEY=your_llama_parse_key
LLM_MODEL=gemini-pro
CHROMA_PERSIST_DIR=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Usage

1. Run the complete pipeline:
```bash
python main.py
```

The system will:
1. Scrape the website for documents
2. Process and ingest documents into ChromaDB
3. Start an interactive query mode

2. Run individual components:
```python
# Web scraping
from scrape import scrape_and_download
scrape_and_download("https://www.arista.com", max_files=100) #change max files as needed

# Document ingestion
from ingestion import DocumentIngestion
ingestion = DocumentIngestion()
ingestion.ingest_documents("./documents")

# RAG queries
from rag import RAGSystem
rag = RAGSystem()
response = rag.query("Your question here")
```

## Configuration

### Web Scraper
- `max_pages`: Maximum pages to crawl
- `max_files`: Maximum files to download
- `valid_extensions`: File types to download

### Document Ingestion
- `CHROMA_PERSIST_DIR`: ChromaDB storage location
- `EMBEDDING_MODEL`: Embedding model for vector storage

### RAG System
- `LLM_MODEL`: Language model to use
- `RELEVANCE_THRESHOLD`: Minimum relevance score for answers

## Project Structure

```
Arista-RAG-Pipeline/
├── main.py              # Main entry point
├── scrape.py            # Web scraping module
├── ingestion.py         # Document processing
├── rag.py              # RAG system implementation
├── documents/          # Downloaded files
├── chroma_db/         # Vector database
├── .env               # Environment variables
├── pyproject.toml     # Poetry dependencies
└── README.md         # This file
```

## Dependencies

- selenium: Web browser automation
- chromadb: Vector database
- llama-index: Document processing
- google-generativeai: Gemini model
- python-dotenv: Environment management
- deepeval: Hallucination detection

look at full list in pyproject.toml file

## Workflow

1. **Web Scraping Phase**
   - Crawls specified website
   - Downloads relevant documents
   - Stores in `documents` directory

2. **Document Ingestion Phase**
   - Processes new documents
   - Extracts text and metadata
   - Stores in ChromaDB

3. **RAG Query Phase**
   - Processes user questions
   - Retrieves relevant context
   - Generates answers using Gemini
   - Validates responses

## Troubleshooting

1. **Web Scraping Issues**
   - Check ChromeDriver version
   - Verify network connectivity
   - Adjust crawling parameters

2. **Document Processing Issues**
   - Verify LlamaParse API key
   - Check document formats
   - Monitor ChromaDB storage

3. **RAG System Issues**
   - Validate Google API key
   - Check model availability
   - Monitor response quality




## Contact & Authors

Neil Thomas
neilthomas@berkeley.edu

Yash Chitambar
yash_chitambar@berkeley.edu

Dhruv Hebbar
dhebbar@berkeley.edu

Hasset Mekuria
hasset_mek@berkeley.edu

Avy Harish
avyukth.harish@berkeley.edu

Jack White
jackwhite@berkeley.edu