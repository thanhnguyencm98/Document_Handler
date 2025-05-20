# Document Management System

A Python application that allows users to upload, summarize, classify, and search documents using Ollama with Llama 3.1.

## Features

- Upload various document types (PDF, Excel, PowerPoint, Word)
- Extract text content from documents
- Generate summaries using Llama 3.1
- Categorize documents automatically
- Extract keywords from documents
- Search documents by content, category, or keywords
- View and download stored documents

## Requirements

- Python 3.7+
- Ollama with Llama 3.1 model installed and running

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

3. Ensure Ollama is installed and the Llama 3.1 model is downloaded:

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Download the Llama 3.1 model
ollama pull llama3.1
```

## Usage

1. Start the Ollama server:

```bash
ollama serve
```

2. In a separate terminal, run the application:

```bash
streamlit run main.py
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
2. **Processing**: The system extracts text from documents and sends it to Llama 3.1 for analysis.
3. **Analysis**: Llama 3.1 generates a summary, category, and keywords for each document.
4. **Storage**: Document information and analysis are stored in a JSON database.
5. **Search**: Users can search for documents based on content, category, or keywords.
6. **View**: Users can view document summaries and download the original files.

## Configuration

By default, the application connects to Ollama running on `http://localhost:11434`. If you need to change this, modify the `api_url` parameter in the `LLMService` initialization in `main.py`.

## Limitations

- The application sends only the first 4000 characters of a document to Llama 3.1 for analysis to keep requests within reasonable sizes.
- Large documents may take time to process, especially for text extraction.
- The quality of summaries and categorization depends on the capabilities of the Llama 3.1 model.

## License

This project is licensed under the MIT License - see the LICENSE file for details.