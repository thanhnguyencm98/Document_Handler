import os
import json
import csv
import zipfile
import io
import pandas as pd
import streamlit as st
from document_storage import DocumentStorage

def export_key_concepts_csv(storage):
    """
    Export key concepts from all documents as CSV.
    
    Args:
        storage: DocumentStorage instance
        
    Returns:
        bytes: CSV file content as bytes
    """
    documents = storage.get_all_documents()
    
    # Prepare data for CSV
    csv_data = []
    for doc in documents:
        for concept in doc.get("key_concepts", []):
            csv_data.append({
                "Document": doc["filename"],
                "Category": doc["category"],
                "Key Concept": concept,
                "Upload Date": doc["upload_date"]
            })
    
    # Convert to DataFrame and then to CSV
    df = pd.DataFrame(csv_data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    return csv_buffer.getvalue().encode()

def export_document_summaries(storage):
    """
    Export document summaries and metadata as JSON.
    
    Args:
        storage: DocumentStorage instance
        
    Returns:
        bytes: JSON file content as bytes
    """
    documents = storage.get_all_documents()
    
    # Create simplified document records with summaries and concepts
    export_data = []
    for doc in documents:
        export_data.append({
            "filename": doc["filename"],
            "category": doc["category"],
            "upload_date": doc["upload_date"],
            "summary": doc["summary"],
            "key_concepts": doc["key_concepts"],
            "keywords": doc["keywords"]
        })
    
    return json.dumps(export_data, indent=2).encode()

def export_document_bulk_zip(storage):
    """
    Create a ZIP archive containing document summaries and key concepts as text files.
    
    Args:
        storage: DocumentStorage instance
        
    Returns:
        bytes: ZIP file content as bytes
    """
    documents = storage.get_all_documents()
    
    # Create a ZIP file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add a summary for each document
        for doc in documents:
            # Create a text file for each document
            filename = f"{doc['id']}_summary.txt"
            content = (
                f"Document: {doc['filename']}\n"
                f"Category: {doc['category']}\n"
                f"Upload Date: {doc['upload_date']}\n\n"
                f"SUMMARY\n{'='*50}\n{doc['summary']}\n\n"
                f"KEY CONCEPTS\n{'='*50}\n"
            )
            
            # Add each key concept as a bullet point
            for concept in doc.get("key_concepts", []):
                content += f"• {concept}\n"
            
            content += f"\nKEYWORDS\n{'='*50}\n"
            content += ", ".join(doc["keywords"])
            
            zip_file.writestr(filename, content)
        
        # Add a CSV with all key concepts
        csv_data = []
        for doc in documents:
            for i, concept in enumerate(doc.get("key_concepts", [])):
                csv_data.append([
                    doc["filename"], 
                    doc["category"],
                    i+1,  # Concept number
                    concept
                ])
        
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow(["Document", "Category", "Concept #", "Key Concept"])
        csv_writer.writerows(csv_data)
        
        zip_file.writestr("all_key_concepts.csv", csv_buffer.getvalue())
    
    return zip_buffer.getvalue()

def display_export_page(storage):
    """
    Page for exporting document data.
    
    Args:
        storage: DocumentStorage instance
    """
    st.header("Export Document Analysis")
    st.write("Export document summaries and key concepts in various formats.")
    
    export_format = st.radio(
        "Select export format:",
        ["Key Concepts (CSV)", "Document Summaries (JSON)", "Complete Export (ZIP)"]
    )
    
    if st.button("Generate Export"):
        with st.spinner("Preparing export..."):
            if export_format == "Key Concepts (CSV)":
                data = export_key_concepts_csv(storage)
                filename = "document_key_concepts.csv"
                mime = "text/csv"
            elif export_format == "Document Summaries (JSON)":
                data = export_document_summaries(storage)
                filename = "document_summaries.json"
                mime = "application/json"
            else:  # ZIP
                data = export_document_bulk_zip(storage)
                filename = "document_analysis_export.zip"
                mime = "application/zip"
            
            st.success(f"Export ready: {filename}")
            
            st.download_button(
                label="Download Export",
                data=data,
                file_name=filename,
                mime=mime
            )