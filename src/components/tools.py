import requests
from duckduckgo_search import DDGS
from tavily import TavilyClient
from src.components.pdf_handler import SessionPDFStore
from src.utils.config import TAVILY_API_KEY
from src.utils.logger import setup_logger

logger = setup_logger("tools")

# Global vectorstore (loaded in main.py)
college_vectorstore = None


def set_college_vectorstore(vs):
    """Set the college vectorstore (called from main.py)"""
    global college_vectorstore
    college_vectorstore = vs


def get_ip_address():
    """Get user's public IP address"""
    try:
        logger.info("Calling IP lookup API...")
        response = requests.get("https://api.ipify.org?format=json", timeout=5)
        ip = response.json()["ip"]
        logger.info(f"✅ IP found: {ip}")
        return ip
    except Exception as e:
        logger.error(f"IP lookup failed: {e}")
        return "Unable to retrieve IP address"


# def search_web(query: str, max_results: int = 3):
#     """Search the web using DuckDuckGo"""
#     try:
#         logger.info(f"Searching web for: '{query}'...")
#         results = DDGS().text(query, max_results=max_results)

#         if not results:
#             return "No results found"
        
#         formatted_results = []
#         for i, result in enumerate(results, 1):
#             formatted_results.append(f"{i}. {result['title']}\n   {result['body'][:200]}...")

#         output = "\n\n".join(formatted_results)
#         logger.info(f"✅ Found {len(results)} results")
#         return output
#     except Exception as e:
#         logger.error(f"Web search failed: {e}")
#         return f"Search failed: {str(e)}"


def search_web(query: str):
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(query, max_results=4)
        
        formatted_results = []
        for i, result in enumerate(response['results'], 1):
            formatted_results.append(
                f"{i}. {result['title']}\n   {result['content']}"
            )
        
        return "\n\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return f"Search failed: {str(e)}"


def get_college_info(question: str):
    """Answer questions about Madras Christian College using RAG"""
    try:
        logger.info("Searching college knowledge base...")

        if college_vectorstore is None:
            return "College knowledge base is not available."
        
        docs = college_vectorstore.similarity_search(query=question, k=3)

        if not docs:
            return "I don't have specific information about that in my college knowledge base."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            text = doc.page_content.strip()
            context_parts.append(f"[Source {i}]\n{text}")
        
        context = "\n\n".join(context_parts)
        logger.info(f"✅ Found {len(docs)} relevant chunks")
        return context
        
    except Exception as e:
        logger.error(f"College info search failed: {e}")
        return "Unable to retrieve college information."


def query_uploaded_pdf(question: str, session_store: SessionPDFStore):
    """Query user's uploaded PDFs"""
    return session_store.query(question)


# Tool definitions for LLM
TOOLS = [
    {"type": "function",
        "function": {"name": "get_ip_address",
                     "description": "Get the user's current public IP address.",
                     "parameters": {"type": "object", "properties": {}, "required": []}}},

    {"type": "function",
        "function": {"name": "search_web",
                    "description": "Search the web for current information, news, or facts.",
                    "parameters": {"type": "object",
                                   "properties": {"query": {"type": "string", "description": "The search query"}},
                                   "required": ["query"]}}},

    {"type": "function",
        "function": {"name": "get_college_info",
                     "description": "Get information about Madras Christian College (MCC) from the knowledge base.",
                     "parameters": {"type": "object",
                                    "properties": {"question": {"type": "string", "description": "The user's question about the college"}},
                                    "required": ["question"]}}},

    {"type": "function",
        "function": {"name": "query_uploaded_pdf",
                     "description": "Answer questions from user's uploaded PDF documents.",
                     "parameters": {"type": "object",
                                    "properties": {"question": {"type": "string", "description": "The question about the PDF content"}},
                                    "required": ["question"]}}}
                                    
]