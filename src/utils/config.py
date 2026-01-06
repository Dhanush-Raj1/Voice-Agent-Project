import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Audio Configuration
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION_MS = 250
MIN_SILENCE_DURATION_MS = 1500
SPEECH_PAD_MS = 300

# PDF Configuration  
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# LLM Configuration
LLM_MODEL = "qwen/qwen3-32b"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 200

# Vector Store
COLLEGE_INDEX_NAME = "voice-agent"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# System Prompt
SYSTEM_PROMPT = """You are a helpful voice assistant named Agent.

Guidelines:
- Keep responses SHORT (2-3 sentences max) since this is voice conversation
- Be conversational and friendly
- When asked about the college, use the get_college_info tool
- When asked about uploaded PDFs, use the query_uploaded_pdf tool
- Provide direct, clear answers without lengthy explanations

You have access to these tools:
1. get_ip_address - Get the user's public IP address
2. search_web - Search the internet for current information
3. get_college_info - Answer questions about Madras Christian College
4. query_uploaded_pdf - Answer questions from user's uploaded PDF documents
"""