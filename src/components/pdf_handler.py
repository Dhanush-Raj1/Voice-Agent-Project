import os
import tempfile
from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.utils.logger import setup_logger

logger = setup_logger("pdf_handler")


class SessionPDFStore:
    """Manages PDF documents for a single WebSocket session"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.uploaded_pdfs: List[Dict] = []
        self.vector_store: Optional[FAISS] = None
        self.all_documents: List[Document] = []
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                            chunk_overlap=CHUNK_OVERLAP,
                                                            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""])
    
    def add_pdf(self, filename: str, pdf_bytes: bytes) -> Dict:
        """Process and add a PDF to the session store"""

        try:
            logger.info(f"Processing PDF: {filename}")
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_path = tmp_file.name
            
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            
            logger.info(f"Loaded {len(pages)} pages")
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(pages)
            
            # Add metadata
            for chunk in chunks:
                chunk.metadata['source_file'] = filename
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # Add to documents
            self.all_documents.extend(chunks)
            
            # Create or update vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                new_store = FAISS.from_documents(chunks, self.embeddings)
                self.vector_store.merge_from(new_store)
            
            # Clean up
            os.unlink(tmp_path)
            
            # Track PDF info
            pdf_info = {
                'filename': filename,
                'chunks': len(chunks),
                'pages': len(pages),
                'status': 'ready'
            }
            self.uploaded_pdfs.append(pdf_info)
            
            logger.info(f"✅ PDF added (Total docs: {len(self.all_documents)})")
            return pdf_info
            
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            return {
                'filename': filename,
                'error': str(e),
                'status': 'failed'
            }
    

    def query(self, question: str, k: int = 3) -> str:
        """Query the uploaded PDFs"""

        try:
            if self.vector_store is None or len(self.all_documents) == 0:
                return "No PDFs have been uploaded yet. Please upload a PDF first."
            
            logger.info(f"Querying PDFs: '{question}'")
            
            docs = self.vector_store.similarity_search(question, k=k)
            
            if not docs:
                return "I couldn't find relevant information in your uploaded PDFs."
            
            # Format context
            context_parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source_file', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                text = doc.page_content.strip()
                context_parts.append(f"[{source} - Page {page}]\n{text}")
            
            context = "\n\n".join(context_parts)
            logger.info(f"✅ Found {len(docs)} relevant chunks")
            return context
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return f"Error querying PDFs: {str(e)}"
    

    def get_pdf_list(self) -> List[Dict]:
        return self.uploaded_pdfs
    
    
    def clear(self):
        self.uploaded_pdfs = []
        self.vector_store = None
        self.all_documents = []
        logger.info("Cleared all session PDFs")