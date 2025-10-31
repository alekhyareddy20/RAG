"""
Vector Database Management Module
----------------------------------
This module handles the creation and management of a Chroma vector database
for document retrieval. It processes PDF documents, splits them into chunks,
generates embeddings, and stores them for efficient similarity search.
"""

import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import nltk
from embed import HuggingFaceEmbeddingFunction

# Download required NLTK data for text processing
nltk.download('punkt')

# Configuration constants
VECTOR_DB_PATH = "chroma"
DOCUMENTS_PATH = "data"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"


def main():
    """
    Main entry point for database setup.
    Handles command-line arguments and orchestrates the database creation process.
    """
    # Parse command-line arguments
    argument_parser = argparse.ArgumentParser(
        description="Build or reset the vector database for document retrieval"
    )
    argument_parser.add_argument(
        "--reset", 
        action="store_true", 
        help="Clear existing database before creating a new one"
    )
    parsed_args = argument_parser.parse_args()
    
    # Clear database if reset flag is provided
    if parsed_args.reset:
        print("✨ Clearing existing database...")
        clear_database()

    # Build or update the vector database
    print("📚 Loading documents...")
    loaded_documents = load_documents()
    
    print("✂️ Splitting documents into chunks...")
    document_chunks = split_documents(loaded_documents)
    
    print("💾 Adding chunks to vector database...")
    add_to_chroma(document_chunks)
    
    print("✅ Database setup complete!")


def load_documents():
    """
    Load all PDF documents from the specified directory.
    
    Returns:
        list[Document]: List of loaded document objects
    """
    pdf_loader = PyPDFDirectoryLoader(DOCUMENTS_PATH)
    return pdf_loader.load()


def split_documents(documents: list[Document]):
    """
    Split documents into smaller chunks for better retrieval accuracy.
    Uses recursive character splitting to maintain context while keeping chunks manageable.
    
    Args:
        documents (list[Document]): List of documents to split
        
    Returns:
        list[Document]: List of document chunks with metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,           # Maximum characters per chunk
        chunk_overlap=80,         # Overlap to maintain context between chunks
        length_function=len,      # Function to measure chunk size
        is_separator_regex=False  # Use simple string separators
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(document_chunks: list[Document]):
    """
    Add document chunks to the Chroma vector database.
    Only adds new chunks that don't already exist in the database.
    
    Args:
        document_chunks (list[Document]): List of document chunks to add
    """
    # Initialize the embedding function for vector generation
    embedding_function = HuggingFaceEmbeddingFunction(
        model_id=EMBEDDING_MODEL_NAME
    )

    # Load or create the vector database
    vector_database = Chroma(
        persist_directory=VECTOR_DB_PATH, 
        embedding_function=embedding_function
    )

    # Generate unique IDs for each chunk
    chunks_with_unique_ids = calculate_chunk_ids(document_chunks)

    # Get existing document IDs to avoid duplicates
    existing_items = vector_database.get(include=[])
    existing_document_ids = set(existing_items["ids"])
    print(f"📊 Number of existing documents in database: {len(existing_document_ids)}")

    # Filter out chunks that already exist in the database
    new_document_chunks = []
    for chunk in chunks_with_unique_ids:
        if chunk.metadata["id"] not in existing_document_ids:
            new_document_chunks.append(chunk)

    # Add new chunks to the database
    if len(new_document_chunks) > 0:
        print(f"➕ Adding {len(new_document_chunks)} new documents to database")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_document_chunks]
        vector_database.add_documents(new_document_chunks, ids=new_chunk_ids)
        vector_database.persist()
    else:
        print("✅ No new documents to add - database is up to date")


def calculate_chunk_ids(document_chunks):
    """
    Generate unique IDs for each document chunk.
    ID format: "data/document.pdf:page_number:chunk_index"
    
    Args:
        document_chunks (list[Document]): Chunks to assign IDs to
        
    Returns:
        list[Document]: Chunks with unique IDs added to metadata
    """
    last_page_identifier = None
    current_chunk_index = 0

    for chunk in document_chunks:
        # Extract source file and page number from metadata
        source_file = chunk.metadata.get("source")
        page_number = chunk.metadata.get("page")
        current_page_identifier = f"{source_file}:{page_number}"

        # Increment chunk index for the same page, reset for new pages
        if current_page_identifier == last_page_identifier:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Create unique chunk ID and add to metadata
        chunk_unique_id = f"{current_page_identifier}:{current_chunk_index}"
        last_page_identifier = current_page_identifier
        chunk.metadata["id"] = chunk_unique_id

    return document_chunks


def clear_database():
    """
    Remove the entire vector database directory.
    Use with caution - this deletes all stored embeddings.
    """
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)
        print("🗑️ Database directory removed")


if __name__ == "__main__":
    main()