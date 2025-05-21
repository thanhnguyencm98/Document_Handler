### Question Answering Feature

The "Ask Questions" feature allows users to:
- Select a specific document or query across all documents
- Ask natural language questions about document content
- Receive AI-generated answers based on the selected documents
- The AI model analyzes the document content and provides accurate answers based only on the information found in the documents# Document Management System

A Python application that allows users to upload, summarize, classify, and search documents using Ollama with various LLM models.

## Features

- Upload various document types (PDF, Excel, PowerPoint, Word)
- Extract text content from documents
- Generate summaries using LLM models
- Categorize documents automatically
- Extract keywords from documents
- Search documents by content, category, or keywords
- View and download stored documents
- Ask questions about document content using AI
- **NEW: Select from different LLM models (Llama 3.1, DeepSeek, Phi3, etc.)**
- **NEW: Delete individual or all documents**

## Requirements

- Python 3.7+
- Ollama with desired LLM models installed and running

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/document-management-system.git
cd document-management-system
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Ensure Ollama is installed and the desired LLM models are downloaded:

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Download models
ollama pull llama3.1
ollama pull deepseek-coder  # Optional
ollama pull phi3            # Optional
ollama pull mistral         # Optional
# Add other models as needed
```

## Usage

1. Start the Ollama server:

```bash
ollama serve
```

2. In a separate terminal, run the application:

```bash
streamlit run main.py
```
or
```bash
python -m streamlit run main.py
```

3. Open your web browser and navigate to http://localhost:8501

## Project Structure

- `main.py`: Main application with Streamlit UI
- `document_processor.py`: Handles document processing workflows
- `document_storage.py`: Manages document storage and retrieval
- `text_extractor.py`: Extracts text from various document formats
- `llm_service.py`: Interface for interacting with Ollama/Llama 3.1

## How It Works

1. **Upload**: Users can upload documents through the web interface.
2. **Processing**: The system extracts text from documents and sends it to the selected LLM model for analysis.
3. **Analysis**: The LLM generates a summary, category, and keywords for each document.
4. **Storage**: Document information and analysis are stored in a JSON database.
5. **Search**: Users can search for documents based on content, category, or keywords.
6. **View**: Users can view document summaries and download the original files.
7. **Ask Questions**: Users can ask questions about document content and get AI-generated answers.
8. **Document Management**: Users can delete individual documents or clear all documents.
9. **Model Selection**: Users can choose different LLM models for processing and question answering.

### New Features

#### Model Selection
- Choose from various LLM models in the sidebar
- Models include Llama 3.1, DeepSeek-Coder, Phi3, Mistral, and more
- Switch models at any time during your session
- Applied to both document analysis and question answering

#### Document Management
- Delete individual documents with confirmation dialog
- "Delete All" option to clear the entire document database
- Document deletion removes both database entries and files from storage

## Configuration

By default, the application connects to Ollama running on `http://localhost:11434`. If you need to change this, modify the `api_url` parameter in the `LLMService` initialization in `main.py`.

The application uses Streamlit's session state to persist model selection between page navigations.

## Limitations

- The application sends only the first 4000 characters of a document to the LLM for analysis to keep requests within reasonable sizes.
- Large documents may take time to process, especially for text extraction.
- The quality of summaries, categorization, and question answering depends on the capabilities of the selected LLM model.
- When querying across all documents, the system currently limits the context to the first 5 documents to avoid token limitations.
- Some models may require more resources than others. If you experience slow performance or errors, try switching to a smaller model.

## License

This project is licensed under the MIT License - see the LICENSE file for details.