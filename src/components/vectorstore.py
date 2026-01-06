import time
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from utils.config import HF_API_KEY, PINECONE_API_KEY, COLLEGE_INDEX_NAME, EMBEDDING_MODEL
from utils.logger import setup_logger

logger = setup_logger("vectorstore")


def build_college_vectorstore(pdf_path: str, index_name: str = COLLEGE_INDEX_NAME):
    """Build and upload college vectorstore to Pinecone"""
    logger.info(f"Building vectorstore from {pdf_path}")
    
    try:
        # Load embeddings
        embeddings = HuggingFaceEndpointEmbeddings(
            model=EMBEDDING_MODEL,
            huggingfacehub_api_token=HF_API_KEY
        )
        
        # Load PDF
        loader = PyPDFLoader(pdf_path, mode="page")
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} pages")
        
        # Create Pinecone index
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(10)
        
        # Upload to Pinecone
        vector_store = PineconeVectorStore.from_documents(
            documents=docs,
            index_name=index_name,
            embedding=embeddings
        )
        
        logger.info(f"✅ Vectorstore created with {len(docs)} documents")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        raise


def load_college_vectorstore(index_name: str = COLLEGE_INDEX_NAME):
    """Load existing college vectorstore from Pinecone"""
    logger.info("Loading college vectorstore...")
    
    try:
        embeddings = HuggingFaceEndpointEmbeddings(
            model=EMBEDDING_MODEL,
            huggingfacehub_api_token=HF_API_KEY
        )
        
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        logger.info(f"✅ Vectorstore loaded (index: {index_name})")
        return vectorstore, embeddings
        
    except Exception as e:
        logger.error(f"Error loading vectorstore: {e}")
        return None, None