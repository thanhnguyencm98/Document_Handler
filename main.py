import os
import streamlit as st
from document_processor import DocumentProcessor
from document_storage import DocumentStorage
from llm_service import LLMService

def main():
    """Main application for document management system."""
    st.set_page_config(page_title="Document Management System", layout="wide")
    st.title("Document Management System")
    
    # Initialize session state for LLM model if it doesn't exist
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = "llama3.1"
    
    # LLM model selection in sidebar
    st.sidebar.title("Settings")
    available_models = ["llama3.1", "llama3.2", "deepseek-r1", "phi4"]
    selected_model = st.sidebar.selectbox("Select LLM Model", available_models, index=available_models.index(st.session_state.llm_model))
    
    # Update session state if model changed
    if selected_model != st.session_state.llm_model:
        st.session_state.llm_model = selected_model
        st.sidebar.success(f"Model changed to {selected_model}")

    # Initialize services
    storage = DocumentStorage("./documents_db.json", "./uploads")
    llm = LLMService(model=st.session_state.llm_model)
    processor = DocumentProcessor(storage, llm)

    # Create directory for uploads if it doesn't exist
    os.makedirs("./uploads", exist_ok=True)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload", "Search", "View Documents", "Ask Questions"])

    if page == "Upload":
        display_upload_page(processor)
    elif page == "Search":
        display_search_page(storage)
    elif page == "View Documents":
        display_documents_page(storage)
    elif page == "Ask Questions":
        display_ask_questions_page(storage, llm)

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
                    
                    st.subheader("Important Points")
                    for concept in doc_info["important_points"]:
                        st.markdown(f"• {concept}")
                    
                    st.subheader("Main sections")
                    for title, brief in zip(doc_info["sections_title"], doc_info["sections_brief"]):
                        st.markdown(f"#### {title}")
                        st.markdown(f"{brief}")

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
                display_document_results(results, storage)
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
    
    # Add a button to delete all documents
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Delete All", type="secondary"):
            st.session_state.confirm_delete_all = True
    
    # Confirmation dialog for delete all
    if st.session_state.get('confirm_delete_all', False):
        st.warning("Are you sure you want to delete ALL documents? This cannot be undone.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, Delete All"):
                for doc in documents:
                    storage.delete_document(doc["id"])
                st.session_state.confirm_delete_all = False
                st.success("All documents deleted successfully!")
                st.rerun()
        with col2:
            if st.button("Cancel"):
                st.session_state.confirm_delete_all = False
                st.rerun()
    
    display_document_results(documents, storage)

def display_ask_questions_page(storage, llm):
    """Page for asking questions about documents."""
    st.header("Ask Questions About Documents")
    st.write("Ask questions about your uploaded documents, and the AI will provide answers based on their content.")
    
    # Get all documents
    all_documents = storage.get_all_documents()
    
    if not all_documents:
        st.info("No documents found. Please upload some documents first.")
        return
    
    # Create document selection dropdown
    doc_options = ["All Documents"] + [doc["filename"] for doc in all_documents]
    selected_doc = st.selectbox("Select document to query", options=doc_options)
    
    # Get user question
    user_question = st.text_input("Enter your question about the document(s)")
    
    # Create columns for the Ask button and model info
    col1, col2 = st.columns([1, 2])
    with col1:
        ask_button = st.button("Ask")
    with col2:
        st.caption(f"Using model: {st.session_state.llm_model}")
    
    if user_question and ask_button:
        with st.spinner("Generating answer..."):
            # Prepare context based on selected document(s)
            if selected_doc == "All Documents":
                # Use all documents as context, but limit to avoid token limits
                context = ""
                for doc in all_documents[:5]:  # Limit to first 5 documents to avoid token limits
                    context += f"\nDocument: {doc['filename']}\nContent: {doc['content']}...\n"
            else:
                # Find the selected document
                for doc in all_documents:
                    if doc["filename"] == selected_doc:
                        context = f"Document: {doc['filename']}\nContent: {doc['content']}\n"
                        break
            
            # Get answer from LLM
            answer = llm.answer_question(user_question, context)
            
            # Display the answer
            st.subheader("Answer")
            st.write(answer)
            
            # Display source information
            st.subheader("Source")
            if selected_doc == "All Documents":
                st.write("Answer based on all available documents")
            else:
                st.write(f"Answer based on: {selected_doc}")

def display_document_results(documents, storage):
    """Display document results in an organized manner."""
    for doc in documents:
        with st.expander(f"{doc['filename']} ({doc['category']})"):
            # Add a delete button
            col1, col2 = st.columns([5, 1])
            with col2:
                if st.button("Delete", key=f"delete_{doc['id']}", type="secondary"):
                    st.session_state[f"confirm_delete_{doc['id']}"] = True
            
            # Confirmation dialog for delete
            if st.session_state.get(f"confirm_delete_{doc['id']}", False):
                st.warning("Are you sure you want to delete this document? This cannot be undone.")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Delete", key=f"confirm_{doc['id']}"):
                        if storage.delete_document(doc["id"]):
                            st.success("Document deleted successfully!")
                            st.session_state[f"confirm_delete_{doc['id']}"] = False
                            st.rerun()
                        else:
                            st.error("Failed to delete document.")
                with col2:
                    if st.button("Cancel", key=f"cancel_{doc['id']}"):
                        st.session_state[f"confirm_delete_{doc['id']}"] = False
                        st.rerun()
            
            st.write(f"**Uploaded:** {doc['upload_date']}")
            st.write(f"**Category:** {doc['category']}")
            st.write(f"**Keywords:** {', '.join(doc['keywords'])}")
            
            st.subheader("Summary")
            st.write(doc["summary"])
            
            st.subheader("Important Points")
            for concept in doc["important_points"]:
                st.markdown(f"• {concept}")

            st.subheader("Main sections")
            for title, brief in zip(doc["sections_title"], doc["sections_brief"]):
                st.markdown(f"#### {title}")
                st.markdown(f"{brief}")

            st.download_button(
                label="Download Document",
                data=open(doc["file_path"], "rb"),
                file_name=doc["filename"],
                mime="application/octet-stream"
            )

if __name__ == "__main__":
    main()