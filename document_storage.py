import os
import json
import re

class DocumentStorage:
    """
    Handles storage and retrieval of document information.
    """
    
    def __init__(self, db_path, upload_dir):
        """
        Initialize the document storage.
        
        Args:
            db_path: Path to the JSON file for document database
            upload_dir: Directory for uploaded document files
        """
        self.db_path = db_path
        self.upload_dir = upload_dir
        
        # Create upload directory if it doesn't exist
        os.makedirs(upload_dir, exist_ok=True)
        
        # Initialize database if it doesn't exist
        if not os.path.exists(db_path):
            with open(db_path, "w") as f:
                json.dump([], f)
    
    def save_document(self, doc_info):
        """
        Save document information to the database.
        
        Args:
            doc_info: Dictionary containing document information
        """
        # Load existing documents
        documents = self._load_documents()
        
        # Add the new document
        documents.append(doc_info)
        
        # Save back to file
        with open(self.db_path, "w") as f:
            json.dump(documents, f, indent=2)
    
    def get_document(self, doc_id):
        """
        Get document information by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            dict: Document information or None if not found
        """
        documents = self._load_documents()
        for doc in documents:
            if doc["id"] == doc_id:
                return doc
        return None
    
    def get_all_documents(self):
        """
        Get all documents.
        
        Returns:
            list: List of document information dictionaries
        """
        return self._load_documents()
    
    def search_by_content(self, query):
        """
        Search documents by content.
        
        Args:
            query: Search query string
            
        Returns:
            list: List of document information dictionaries matching the query
        """
        documents = self._load_documents()
        results = []
        
        query_terms = query.lower().split()
        
        for doc in documents:
            content = doc["content"].lower()
            summary = doc["summary"].lower()
            
            # Check if all query terms appear in content or summary
            if all(term in content or term in summary for term in query_terms):
                results.append(doc)
        
        return results
    
    def search_by_category(self, category):
        """
        Search documents by category.
        
        Args:
            category: Category string
            
        Returns:
            list: List of document information dictionaries matching the category
        """
        documents = self._load_documents()
        results = []
        
        category_lower = category.lower()
        
        for doc in documents:
            doc_category = doc["category"].lower()
            
            # Partial match on category
            if category_lower in doc_category:
                results.append(doc)
        
        return results
    
    def search_by_keyword(self, keyword):
        """
        Search documents by keyword.
        
        Args:
            keyword: Keyword string
            
        Returns:
            list: List of document information dictionaries matching the keyword
        """
        documents = self._load_documents()
        results = []
        
        keyword_lower = keyword.lower()
        
        for doc in documents:
            # Check if keyword appears in any of the document keywords
            if any(keyword_lower in kw.lower() for kw in doc["keywords"]):
                results.append(doc)
        
        return results
    
    def delete_document(self, doc_id):
        """
        Delete a document from storage.
        
        Args:
            doc_id: Document ID
            
        Returns:
            bool: True if document was deleted, False otherwise
        """
        documents = self._load_documents()
        
        for i, doc in enumerate(documents):
            if doc["id"] == doc_id:
                # Remove file from disk
                try:
                    os.remove(doc["file_path"])
                except OSError:
                    # Log the error but continue
                    print(f"Error removing file: {doc['file_path']}")
                
                # Remove from database
                documents.pop(i)
                
                # Save updated database
                with open(self.db_path, "w") as f:
                    json.dump(documents, f, indent=2)
                
                return True
        
        return False
    
    def _load_documents(self):
        """
        Load documents from the database file.
        
        Returns:
            list: List of document information dictionaries
        """
        try:
            with open(self.db_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Return empty list if file doesn't exist or is invalid
            return []