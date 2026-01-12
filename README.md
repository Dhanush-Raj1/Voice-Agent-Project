<h1 align="center">🎤 AI Voice Agent with RAG & PDF Intelligence</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=black&labelColor=white&color=FFD43B" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=black&labelColor=white&color=009688" />
  <img src="https://img.shields.io/badge/WebSocket-010101?style=for-the-badge&logo=Socket.io&logoColor=black&labelColor=white&color=purple" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=LangChain&logoColor=black&labelColor=white&color=1C3C3C" />
  <img src="https://img.shields.io/badge/Groq-234452?style=for-the-badge&logoColor=black&labelColor=white&color=f4a852" />
  <img src="https://img.shields.io/badge/Pinecone-234452?style=for-the-badge&logoColor=black&labelColor=green&color=cyan" />
  <img src="https://img.shields.io/badge/Whisper-412991?style=for-the-badge&logoColor=black&labelColor=white&color=412991" />
  <img src="https://img.shields.io/badge/RAG-234452?style=for-the-badge&logoColor=black&labelColor=white&color=yellow" />
  <img src="https://img.shields.io/badge/Edge_TTS-0078D4?style=for-the-badge&logoColor=black&labelColor=white&color=0078D4" />
</p>

<h3 align="center">Real-time AI-powered Voice Assistant with Document Intelligence and Web Search</h3>  
<h3 align="center">Talk to Your PDFs, Search the Web, and Get Instant Answers - All by Voice!</h3>  

<br>

# 🚀 Live Application
🌐 The application is deployed and live
  
👉 [Access the web app here](https://voice-agent-project-eszo.onrender.com)    
<br>
> [!NOTE]
> The voice agent requires lot of computing power as a result the agent might be slow or takes time to respond.  
> For best experience, please use **Google Chrome**, as the application has been tested primarily on Chrome.       
    
> [!TIP]  
> For the best experience, please refer to the [Usage Guide](#-usage-guide) section below to learn how to navigate and use the web app effectively.


<br>

# 🎯 Project Overview

A **real-time voice agent** that combines speech recognition, natural language understanding, and text-to-speech synthesis to create an intelligent conversational assistant. The system leverages **Retrieval-Augmented Generation (RAG)** to answer questions from uploaded PDFs, performs web searches, and finds your public IP address - all through natural voice interactions.

### Key Capabilities

**🎙️ Voice Processing**
- Real-time speech detection using **Silero VAD** (Voice Activity Detection)
- Automatic transcription with **Whisper AI** via Groq
- Natural voice responses using **Edge TTS**
- Continuous audio streaming over WebSocket

**📚 Document Intelligence**
- Upload and query multiple PDF documents
- Smart chunking and semantic search with **FAISS**
- Session-based document memory
- Contextual answers from your documents

**🔍 Web Search & Knowledge Retrieval**
- Real-time web search using **Tavily API**
- Pre-loaded knowledge base about Madras Christian College - RAG 
- IP address lookup capability
- Multi-tool orchestration with intelligent routing

**🤖 Advanced AI Features**
- Powered by **Llama 3.3 70B** and **Qwen 3 32B** models
- Two-stage reasoning: tool selection + response generation
- Context-aware conversation handling
- Hallucination filtering for transcription accuracy

<br>

# 🚀 Features

- ✅ **Real-time Voice Interaction**: Speak naturally and receive instant voice responses
- ✅ **PDF Upload & Query**: Upload documents and ask questions about their content
- ✅ **Web Search Integration**: Get current information from the internet
- ✅ **College Knowledge Base**: Pre-loaded information about Madras Christian College- RAG 
- ✅ **Session Memory**: Each session maintains its own conversation context
- ✅ **Smart Voice Detection**: Automatic speech start/end detection
- ✅ **Multi-tool Orchestration**: Intelligently routes queries to appropriate tools
- ✅ **Clean UI**: Modern, responsive web interface with real-time status updates
- ✅ **Efficient Processing**: Optimized audio streaming and processing pipeline

<br>

# 🏗️ Architecture

```
┌─────────────┐
│   Browser   │
│  (WebRTC)   │
└──────┬──────┘
       │ Audio Stream (PCM 16kHz)
       ↓
┌─────────────────────────────────┐
│      FastAPI + WebSocket        │
│                                 │
│  ┌──────────────────────────┐   │
│  │   Silero VAD Processor   │   │
│  │  (Speech Detection)      │   │
│  └────────┬─────────────────┘   │
│           │ Speech Segments     │
│           ↓                     │
│  ┌──────────────────────────┐   │
│  │  Whisper STT (Groq API)  │   │
│  │  (Speech → Text)         │   │
│  └────────┬─────────────────┘   │
│           │ Transcript          │
│           ↓                     │
│  ┌──────────────────────────┐   │
│  │   LLM Agent              │   │
│  │   (Qwen 3 32B Tool Call) │   │
│  │   (Llama 3.3 70B Final)  │   │
│  └────────┬─────────────────┘   │
│           │                     │
│     ┌─────┴─────┐               │
│     │   Tools   │               │
│     │           │               │
│  ┌──┴───────────┴──┐            │
│  │ • Tavily Search │            │
│  │ • PDF Query     │            │
│  │ • College Info  │            │
│  │ • IP Lookup     │            │
│  └──┬───────────┬──┘            │
│     │           │               │
│     ↓           ↓               │
│  ┌─────────┐ ┌──────────────┐   │
│  │Pinecone │ │ FAISS        │   │
│  │Vector DB│ │(Session PDFs)│   │
│  └─────────┘ └──────────────┘   │
│                                 │
│  ┌──────────────────────────┐   │
│  │    Edge TTS Engine       │   │
│  │    (Text → Speech)       │   │
│  └────────┬─────────────────┘   │ 
│           │ Audio (Base64)      │
└───────────┼─────────────────────┘
            ↓
┌─────────────────────────────────┐
│   Browser Audio Playback        │
└─────────────────────────────────┘
```

<br>

# 🗂️ Project Structure

```
📂 Voice-Agent-Project
│
├── 📂 src
│   ├── main.py                      # FastAPI server & WebSocket handler
│   │
│   ├── 📂 components
│   │   ├── agent.py                  # LLM agent with tool orchestration
│   │   ├── audio.py                  # VAD processor & Whisper STT
│   │   ├── tts.py                    # Edge TTS for voice synthesis
│   │   ├── tools.py                  # Tool definitions & implementations
│   │   ├── pdf_handler.py            # PDF processing & FAISS vector store
│   │   ├── vectorstore.py            # Pinecone integration
│   │   └── websocket.py              # WebSocket connection handler
│   │
│   └── 📂 utils
│       ├── config.py                 # Configuration & environment variables
│       └── logger.py                 # Logging setup
│
├── 📂 templates
│   └── index.html                    # Web interface (HTML/CSS/JS)
│
├── 📂 data
│   └── college_info.pdf              # Pdf file for RAG
│   └── tesla_report.pdf              # Pdf file for upload and chat with 
│
├── .gitignore
├── requirements.txt
└── README.md
```

<br>

# 🛠️ Tech Stack

**Backend Framework**
- **FastAPI** - High-performance async web framework
- **WebSocket** - Real-time bidirectional communication
- **Uvicorn** - ASGI server

**AI & ML Models**
- **Groq API** - Fast LLM inference (Llama 3.3 70B, Qwen 3 32B)
- **Whisper Large V3 Turbo** - Speech-to-text transcription
- **Silero VAD** - Voice activity detection
- **Edge TTS** - Text-to-speech synthesis

**Document Processing**
- **LangChain** - LLM application framework
- **PyPDFLoader** - PDF document loading
- **FAISS** - Vector similarity search (session PDFs)
- **Pinecone** - Cloud vector database (college knowledge)
- **HuggingFace Embeddings** - Text embeddings (MiniLM-L6-v2)

**Search & Tools**
- **Tavily API** - Web search for AI agents
- **IPify API** - IP address lookup

**Frontend**
- **HTML5** - Structure
- **CSS3** - Modern styling with gradients & animations
- **Vanilla JavaScript** - WebSocket client & audio handling

<br>

# 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- Conda (recommended) or virtualenv
- API Keys for: Groq, Pinecone, HuggingFace, Tavily

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/voice-agent-project.git
cd voice-agent-project
```

### 2️⃣ Create Virtual Environment
```bash
# Using Conda
conda create -p envi python==3.9 -y
conda activate envi

# OR using venv
python -m venv envi
source envi/bin/activate  # On macOS/Linux
envi\Scripts\activate     # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables
Create a `.env` file in the root directory:


### 5️⃣ Build the College Vector Store (One-time Setup)
```python
# Run this once to create the Pinecone index
from src.components.vectorstore import build_college_vectorstore

build_college_vectorstore(
    pdf_path="data/college_info.pdf",
    index_name="voice-agent"
)
```

### 6️⃣ Run the Application
```bash
python src/main1.py
```

The server will start at: **`http://localhost:8000`**

<br>

# 🌐 Usage Guide

👉 [Access the web app](https://voice-agent-project-eszo.onrender.com)

1. **Open in Chrome**: Launch the application using **Google Chrome**, which is the primary tested browser
2. **Start Recording**: Click the **"Start Recording"** button to connect
3. **Grant Microphone Access**: Allow browser to access your microphone
4. **Start Talking**: The system automatically detects when you speak
5. **Response**: Once you stop speaking the response is then generated 

### 💬 Example Interactions

**General Conversation**
- "Hello, how are you?"
- "What can you help me with?"

**IP Address Lookup**
- "What's my current IP address?"
- "Tell me my IP"

**Web Search**
- "Search for the latest AI news"
- "Who is the current president of India?"
- "What are the latest advancements in machine learning?"

**College Knowledge Base**
- "Tell me about MCC"
- "What courses are offered by Madras Christian College?"
- "Tell me about the undergraduate programs at MCC"

**PDF Upload & Query**
- Click **"Upload PDF"** button to upload documents
- "Summarize the PDF I just uploaded"
- "What is the revenue mentioned in the Tesla report?"
- "From the uploaded PDF, tell me about [specific topic]"

**Stop Recording**
- Click **"Stop Recording"** to finalize the current speech and disconnect the websocket

### 🎯 Tips for Best Experience

- ✅ Speak clearly and at a normal pace
- ✅ Wait for the agent to finish responding before asking the next question
- ✅ Use the "Stop Recording" button if the system doesn't auto-detect speech 
- ✅ Keep questions concise for faster processing

<br>

## 📊 How It Works

### Voice Processing Pipeline

1. **Audio Capture**: Browser captures microphone input → converts to 16-bit PCM @ 16kHz
2. **VAD Processing**: Silero VAD detects speech frames (512 samples each)
3. **Speech Segmentation**: Collects speech until silence threshold is met
4. **Transcription**: Whisper AI converts audio segment to text
5. **Hallucination Filter**: Removes common false transcriptions

### Agent Reasoning Flow

1. **Tool Selection** (Qwen 3 32B): Analyzes user query → selects appropriate tool(s)
2. **Tool Execution**: Executes selected tools in parallel if needed
3. **Final Response** (Llama 3.3 70B): Synthesizes tool results into natural language
4. **TTS Generation**: Edge TTS converts response text to natural speech
5. **Audio Playback**: Browser plays the audio response

### RAG Implementation

**College Knowledge Base (Pinecone)**
- Pre-loaded PDF about Madras Christian College
- Persistent vector store shared across all sessions
- Fast semantic search for college-related queries

**Session PDFs (FAISS)**
- User-uploaded documents per WebSocket session
- In-memory vector store for fast retrieval
- Automatically cleared when session ends

<br>

# 🔍 Key Components Explained

### Silero VAD Processor
- Processes audio in 512-sample frames
- Calculates speech probability for each frame
- Automatically starts/ends speech detection
- Filters out silence and background noise

### Two-Stage LLM Architecture
**Stage 1: Reasoning (Qwen 3 32B)**
- Fast tool selection and parameter extraction
- Lower latency for decision making
- Efficient token usage

**Stage 2: Response (Llama 3.3 70B)**
- High-quality natural language generation
- Better context understanding
- More coherent and detailed responses

### Session-Based PDF Store
- Each WebSocket connection has isolated PDF storage
- Prevents data leakage between users
- Automatic cleanup on disconnect
- Supports multiple PDFs per session

<br>

# 🎯 Future Enhancements

- [ ] Multi-language support for international users
- [ ] Conversation history persistence with database
- [ ] User authentication and profile management
- [ ] Support for more document formats (DOCX, TXT, etc.)
- [ ] Advanced voice controls (interrupt, replay, etc.)
- [ ] Custom wake word detection
- [ ] Mobile app version (iOS/Android)
- [ ] Integration with calendar and email
- [ ] Multi-modal support (images, charts, etc.)
- [ ] Fine-tuned voice cloning
- [ ] Offline mode with local models

<br>

---

<p align="center">⭐ Star this repo if you find it helpful!</p>
