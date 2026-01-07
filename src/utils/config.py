import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Audio Configuration
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION_MS = 250
MIN_SILENCE_DURATION_MS = 1500
SPEECH_PAD_MS = 300

# PDF Configuration  
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# LLM Model for reasoning 
LLM_MODEL_RES = "qwen/qwen3-32b"
LLM_TEMP_RES = 0.4
LLM_MAX_TOKENS_RES = 500
                       
# LLM Model for final response                       
LLM_MODEL_FIN = "llama-3.3-70b-versatile"   #"openai/gpt-oss-120b"
LLM_TEMP_FIN = 0.5
LLM_MAX_TOKENS_FIN = 150


# Vector Store
COLLEGE_INDEX_NAME = "voice-agent"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# System Prompt
SYSTEM_PROMPT = """You are a helpful voice assistant.

Guidelines:
- Keep responses SHORT (2-3 sentences max) since this is voice conversation
- Be conversational and friendly
- NO emojis or special characters
- Be conversational and friendly
- When using tools, present results clearly and concisely

You have access to these tools:
1. get_ip_address - Get the user's public IP address
2. search_web - Search the internet for current information
3. get_college_info - Answer questions about Madras Christian College
4. query_uploaded_pdf - Answer questions from user's uploaded PDF documents
"""