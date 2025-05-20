import os
import streamlit as st
from document_processor import DocumentProcessor
from document_storage import DocumentStorage
from llm_service import LLMService
from documents_export import display_export_page

def main():
    """Main application for document management system."""
    st.set_page_config(page_title="Document Management System", layout="wide")
    st.title("Document Management System")

    # Initialize services
    storage = DocumentStorage("./documents_db.json", "./uploads")
    llm = LLMService()
    processor = DocumentProcessor(storage, llm)

    # Create directory for uploads if it doesn't exist
    os.makedirs("./uploads", exist_ok=True)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload", "Search", "View Documents", "Export"])

    if page == "Upload":
        display_upload_page(processor)
    elif page == "Search":
        display_search_page(storage)
    elif page == "View Documents":
        display_documents_page(storage)
    elif page == "Export":
        display_export_page(storage)

def display_upload_page(processor):
    """Page for document upload and processing."""
    st.header("Upload Documents")
    st.write("Upload PDF, Excel, PowerPoint, or Word documents to be processed.")
    
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "xlsx", "xls", "pptx", "docx"],
        help="Supported formats: PDF, Excel, PowerPoint, Word"
    )
    
    if uploaded_file:
        st.info(f"Processing: {uploaded_file.name}")
        
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    doc_info = processor.process_document(uploaded_file)
                    
                    st.success("Document processed successfully!")
                    
                    # Display document information
                    st.subheader("Document Summary")
                    st.write(doc_info["summary"])
                    
                    st.subheader("Document Classification")
                    st.write(f"Category: {doc_info['category']}")
                    
                    st.subheader("Key Concepts")
                    for concept in doc_info["key_concepts"]:
                        st.markdown(f"• {concept}")
                    
                    st.subheader("Keywords")
                    st.write(", ".join(doc_info["keywords"]))
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

def display_search_page(storage):
    """Page for searching documents."""
    st.header("Search Documents")
    
    search_query = st.text_input("Enter search query")
    search_type = st.radio("Search by", ["Content", "Category", "Keyword", "Concept"])
    
    if search_query and st.button("Search"):
        with st.spinner("Searching..."):
            if search_type == "Content":
                results = storage.search_by_content(search_query)
            elif search_type == "Category":
                results = storage.search_by_category(search_query)
            elif search_type == "Concept":
                results = storage.search_by_concept(search_query)
            else:  # Keyword
                results = storage.search_by_keyword(search_query)
            
            if results:
                st.success(f"Found {len(results)} document(s).")
                display_document_results(results)
            else:
                st.info("No documents found matching your search.")

def display_documents_page(storage):
    """Page for viewing all documents."""
    st.header("View All Documents")
    
    documents = storage.get_all_documents()
    
    if not documents:
        st.info("No documents found in the system.")
        return
    
    st.success(f"Found {len(documents)} document(s).")
    display_document_results(documents)

def display_document_results(documents):
    """Display document results in an organized manner."""
    for doc in documents:
        with st.expander(f"{doc['filename']} ({doc['category']})"):
            st.write(f"**Uploaded:** {doc['upload_date']}")
            st.write(f"**Category:** {doc['category']}")
            st.write(f"**Keywords:** {', '.join(doc['keywords'])}")
            
            st.subheader("Summary")
            st.write(doc["summary"])
            
            st.subheader("Key Concepts")
            for concept in doc["key_concepts"]:
                st.markdown(f"• {concept}")
            
            st.download_button(
                label="Download Document",
                data=open(doc["file_path"], "rb"),
                file_name=doc["filename"],
                mime="application/octet-stream"
            )

if __name__ == "__main__":
    main()