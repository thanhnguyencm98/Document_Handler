import os
import uuid
import datetime
from text_extractor import TextExtractor

class DocumentProcessor:
    """
    Handles document processing, including extraction, analysis, and storage.
    """
    
    def __init__(self, storage, llm_service):
        """
        Initialize the document processor.
        
        Args:
            storage: DocumentStorage instance for storing document information
            llm_service: LLMService instance for analyzing document content
        """
        self.storage = storage
        self.llm_service = llm_service
        self.extractor = TextExtractor()
    
    def process_document(self, uploaded_file):
        """
        Process an uploaded document file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            dict: Document information including metadata and analysis
        """
        # Generate a unique ID for the document
        doc_id = str(uuid.uuid4())
        
        # Save the file
        file_path = os.path.join("./uploads", f"{doc_id}_{uploaded_file.name}")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text from the document
        text = self.extractor.extract_text(file_path)
        
        if not text:
            raise ValueError("Could not extract text from document")
        
        # Get document analysis from LLM
        analysis = self.llm_service.analyze_document(text)
        
        # Create document info
        doc_info = {
            "id": doc_id,
            "filename": uploaded_file.name,
            "file_path": file_path,
            "upload_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_type": uploaded_file.type,
            "file_size": os.path.getsize(file_path),
            "content": text,
            "summary": analysis["summary"],
            "category": analysis["category"],
            "important_points": analysis["important_points"],
            "sections_title": analysis["sections_title"],
            "sections_brief": analysis["sections_brief"],
            "keywords": analysis["keywords"]
        }
        
        # Store document info
        self.storage.save_document(doc_info)
        
        return doc_info