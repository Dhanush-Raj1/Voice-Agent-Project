import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket
import io
import os
from groq import Groq
import requests  
import json 
from duckduckgo_search import DDGS
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv 
import torch
import base64
import tempfile
from typing import List, Dict, Optional

load_dotenv()

app = FastAPI()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# VAD Configuration
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION_MS = 250
MIN_SILENCE_DURATION_MS = 1500
SPEECH_PAD_MS = 300

# PDF Configuration
CHUNK_SIZE = 800  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

system_prompt = """You are a helpful voice assistant named Agent.

Guidelines:
- Keep responses SHORT (2-3 sentences max) since this is voice conversation
- Be conversational and friendly
- When asked about the college, use the get_college_info tool to get accurate information
- When asked about uploaded PDFs, use the query_uploaded_pdf tool
- Provide direct, clear answers without lengthy explanations
- If you use a tool, briefly mention what you found

You have access to these tools:
1. get_ip_address - Get the user's public IP address
2. search_web - Search the internet for current information
3. get_college_info - Answer questions about Madras Christian College using knowledge base
4. query_uploaded_pdf - Answer questions from user's uploaded PDF documents

Example:
User: "What is Python?"
Bad: "Python is a high-level, interpreted programming language that was created by Guido van Rossum and first released in 1991..."
Good: "Python is a popular programming language known for being easy to learn. It's widely used for web development, data science, and automation."

User: "What's in page 5 of my document?"
Good: "Let me check your document... [uses query_uploaded_pdf tool]"
"""

print("📄 Loading vector store...")

try:
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_API_KEY")
    )
    
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name="voice-agent",
        embedding=embeddings
    )
    
    print("✅ Vector store loaded successfully!")
    print(f"   Index: voice-agent")
    print(f"   Embedding model: sentence-transformers/all-MiniLM-L6-v2")
    
except Exception as e:
    print(f"❌ Error loading vector store: {e}")
    print("   Make sure you've run the setup script first!")
    vectorstore = None


# Initialize Silero VAD
print("\n🎙️ Loading Silero VAD model...")
try:
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    
    (get_speech_timestamps, _, read_audio, *_) = utils
    
    print("✅ Silero VAD loaded successfully!")
    print(f"   Threshold: {VAD_THRESHOLD}")
    print(f"   Min speech: {MIN_SPEECH_DURATION_MS}ms")
    print(f"   Min silence: {MIN_SILENCE_DURATION_MS}ms")
    
except Exception as e:
    print(f"❌ Silero VAD loading failed: {e}")
    model = None
    get_speech_timestamps = None


# Session Storage for PDFs
class SessionPDFStore:
    """Manages PDF documents for a single WebSocket session"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.uploaded_pdfs: List[Dict] = []  # [{filename, chunks, status}]
        self.vector_store: Optional[FAISS] = None
        self.all_documents: List[Document] = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def add_pdf(self, filename: str, pdf_bytes: bytes) -> Dict:
        """Process and add a PDF to the session store"""
        try:
            print(f"\n📄 Processing PDF: {filename}")
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_path = tmp_file.name
            
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            
            print(f"   📖 Loaded {len(pages)} pages")
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(pages)
            
            # Add metadata
            for chunk in chunks:
                chunk.metadata['source_file'] = filename
            
            print(f"   ✂️ Created {len(chunks)} chunks")
            
            # Add to documents
            self.all_documents.extend(chunks)
            
            # Create or update vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                # Add new documents to existing store
                new_store = FAISS.from_documents(chunks, self.embeddings)
                self.vector_store.merge_from(new_store)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            # Track PDF info
            pdf_info = {
                'filename': filename,
                'chunks': len(chunks),
                'pages': len(pages),
                'status': 'ready'
            }
            self.uploaded_pdfs.append(pdf_info)
            
            print(f"   ✅ PDF added to vector store")
            print(f"   📊 Total documents: {len(self.all_documents)}")
            
            return pdf_info
            
        except Exception as e:
            print(f"   ❌ PDF processing error: {e}")
            import traceback
            traceback.print_exc()
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
            
            print(f"\n🔍 Querying PDFs: '{question}'")
            
            # Search for relevant chunks
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
            
            print(f"   ✅ Found {len(docs)} relevant chunks")
            return context
            
        except Exception as e:
            print(f"   ❌ Query error: {e}")
            return f"Error querying PDFs: {str(e)}"
    
    def get_pdf_list(self) -> List[Dict]:
        """Get list of uploaded PDFs"""
        return self.uploaded_pdfs
    
    def clear(self):
        """Clear all PDFs"""
        self.uploaded_pdfs = []
        self.vector_store = None
        self.all_documents = []
        print("🗑️ Cleared all session PDFs")


class SileroVADProcessor:
    """Handles Voice Activity Detection using Silero VAD"""
    
    def __init__(self, model, utils_func):
        self.model = model
        self.get_speech_timestamps = utils_func
        self.reset()
    
    def reset(self):
        """Reset VAD state"""
        self.audio_buffer = []
        self.speech_buffer = []
        self.is_speech_active = False
        self.last_speech_prob = 0.0
        self.silence_frames = 0
        self.speech_frames = 0
        
    def add_audio(self, audio_bytes):
        """Add audio data and return speech segments when detected"""
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        self.audio_buffer.extend(audio_float32)
        
        segments = []
        frame_size = 512
        
        while len(self.audio_buffer) >= frame_size:
            frame = self.audio_buffer[:frame_size]
            self.audio_buffer = self.audio_buffer[frame_size:]
            
            try:
                audio_tensor = torch.from_numpy(np.array(frame, dtype=np.float32))
                speech_prob = self.model(audio_tensor, SAMPLE_RATE).item()
                self.last_speech_prob = speech_prob
                
                if speech_prob >= VAD_THRESHOLD:
                    self.speech_frames += 1
                    self.silence_frames = 0
                    
                    if not self.is_speech_active:
                        self.is_speech_active = True
                        print(f"   🗣️ Speech started (prob: {speech_prob:.3f})")
                    
                    self.speech_buffer.extend(frame)
                    
                else:
                    if self.is_speech_active:
                        self.silence_frames += 1
                        self.speech_buffer.extend(frame)
                        
                        speech_duration_ms = (self.speech_frames * frame_size / SAMPLE_RATE) * 1000
                        silence_duration_ms = (self.silence_frames * frame_size / SAMPLE_RATE) * 1000
                        
                        if (silence_duration_ms >= MIN_SILENCE_DURATION_MS and 
                            speech_duration_ms >= MIN_SPEECH_DURATION_MS):
                            
                            print(f"   🔇 Speech ended (duration: {speech_duration_ms:.0f}ms)")
                            
                            speech_int16 = (np.array(self.speech_buffer) * 32768.0).astype(np.int16)
                            segment_bytes = speech_int16.tobytes()
                            
                            segments.append(segment_bytes)
                            
                            self.speech_buffer = []
                            self.is_speech_active = False
                            self.silence_frames = 0
                            self.speech_frames = 0
                    else:
                        self.silence_frames += 1
                    
            except Exception as e:
                print(f"   ⚠️ VAD processing error: {e}")
        
        return segments, self.last_speech_prob
    
    def finalize(self):
        """Force end current speech segment if active"""
        if self.is_speech_active and len(self.speech_buffer) > 0:
            speech_duration_ms = (len(self.speech_buffer) / SAMPLE_RATE) * 1000
            
            if speech_duration_ms >= MIN_SPEECH_DURATION_MS:
                print(f"   🏁 Finalizing speech segment ({speech_duration_ms:.0f}ms)")
                
                speech_int16 = (np.array(self.speech_buffer) * 32768.0).astype(np.int16)
                segment_bytes = speech_int16.tobytes()
                
                self.reset()
                return segment_bytes
        
        self.reset()
        return None


def get_ip_address():
    """Get user's public IP address"""
    try: 
        print("🌐 Calling IP lookup API...")
        response = requests.get("https://api.ipify.org?format=json", timeout=5)
        ip = response.json()["ip"]
        print(f"   ✅ IP found: {ip}")
        return ip
    except Exception as e:
        print(f"   ❌ IP lookup failed: {e}")
        return "Unable to retrieve IP address"


def search_web(query: str, max_results: int = 3): 
    """Search the web using DuckDuckGo"""
    try: 
        print(f"   🔎 Searching web for: '{query}'...")
        results = DDGS().text(query, max_results=max_results)

        if not results: 
            return "No results found" 
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(f"{i}. {result['title']}\n   {result['body'][:200]}...")

        output = "\n\n".join(formatted_results)
        print(f"   ✅ Found {len(results)} results")
        return output
    except Exception as e: 
        print(f"   ❌ Web search failed: {e}")
        return f"Search failed: {str(e)}"


def get_college_info(question: str):
    """Answer questions about Madras Christian College using RAG"""
    try:
        print(f"   🎓 Searching college knowledge base...")

        if vectorstore is None:
            return "College knowledge base is not available."
        
        docs = vectorstore.similarity_search(query=question, k=3)

        if not docs:
            return "I don't have specific information about that in my college knowledge base."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            text = doc.page_content.strip()
            context_parts.append(f"[Source {i}]\n{text}")
        
        context = "\n\n".join(context_parts)
        print(f"   ✅ Found {len(docs)} relevant chunks")
        return context
        
    except Exception as e:
        print(f"   ❌ College info search failed: {e}")
        return "Unable to retrieve college information."


def query_uploaded_pdf(question: str, session_store: SessionPDFStore):
    """Query user's uploaded PDFs"""
    return session_store.query(question)


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_ip_address",
            "description": "Get the user's current public IP address.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }, 
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information, news, or facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_college_info",
            "description": "Get information about Madras Christian College (MCC) from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question about the college"
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_uploaded_pdf",
            "description": "Answer questions from user's uploaded PDF documents. Use this when the user asks about their documents, files, or uploaded PDFs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question about the PDF content"
                    }
                },
                "required": ["question"]
            }
        }
    }
]


def transcribe_audio(audio_bytes):
    """Convert audio bytes to transcript using Groq"""
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    audio_np /= 32768.0
    
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, SAMPLE_RATE, format='WAV')
    buffer.seek(0)
    buffer.name = "audio.wav"
    
    transcription = groq_client.audio.transcriptions.create(
        file=buffer,
        model="whisper-large-v3-turbo",
        language="en",
        temperature=0.0,
        response_format="text"
    )
    return transcription.strip()


def process_with_agent(user_text, session_store: SessionPDFStore): 
    """Process user text with LLM that can use tools"""
    try: 
        print(f"\n🤖 Agent processing user text...")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        response = groq_client.chat.completions.create(
            messages=messages,
            model="qwen/qwen3-32b",
            tools=tools,
            tool_choice="auto",
            temperature=0.4,
            max_tokens=500,
            stream=False,
            reasoning_format="hidden",   
        )

        response_message = response.choices[0].message

        if response_message.tool_calls: 
            print(f"   🔧 LLM wants to use {len(response_message.tool_calls)} tool(s)")

            messages.append(response_message)

            for tool_call in response_message.tool_calls: 
                function_name = tool_call.function.name 
                function_args = json.loads(tool_call.function.arguments)

                print(f"   📞 Calling tool: {function_name}")

                if function_name == "get_ip_address":
                    function_response = get_ip_address()
                elif function_name == "search_web":
                    query = function_args.get("query", "")
                    function_response = search_web(query)
                elif function_name == "get_college_info":
                    question = function_args.get("question", "")
                    function_response = get_college_info(question)
                elif function_name == "query_uploaded_pdf":
                    question = function_args.get("question", "")
                    function_response = query_uploaded_pdf(question, session_store)
                else: 
                    function_response = f"Unknown tool: {function_name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })

            print("   💬 Getting final response from LLM...")
            final_response = groq_client.chat.completions.create(
                messages=messages,
                model="qwen/qwen3-32b",
                temperature=0.7,
                max_tokens=200
            )

            agent_response = final_response.choices[0].message.content
        else: 
            print("   💬 No tools needed, responding directly")
            agent_response = response_message.content
        
        print(f"   ✅ Response: '{agent_response}'")
        return agent_response
    
    except Exception as e:
        print(f"❌ Agent Error: {e}")
        import traceback
        traceback.print_exc()
        return "Sorry I encountered an error processing your request."


@app.websocket("/ws/audio")
async def websocket_audio(ws: WebSocket):
    await ws.accept()
    print("\n" + "="*60)
    print("🔌 WebSocket Connected")
    print("="*60)

    # Create session-specific stores
    if model and get_speech_timestamps:
        vad_processor = SileroVADProcessor(model, get_speech_timestamps)
    else:
        vad_processor = None
    
    pdf_store = SessionPDFStore(embeddings)
    segment_count = 0

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if msg["type"] == "websocket.receive":
                # Handle JSON messages (PDF upload, commands)
                if "text" in msg:
                    try:
                        data = json.loads(msg["text"])
                        
                        # Handle PDF upload
                        if data.get("type") == "upload_pdf":
                            filename = data.get("filename")
                            pdf_base64 = data.get("data")
                            
                            print(f"\n📤 Receiving PDF upload: {filename}")
                            
                            # Decode base64
                            pdf_bytes = base64.b64decode(pdf_base64)
                            
                            # Process PDF
                            result = pdf_store.add_pdf(filename, pdf_bytes)
                            
                            # Send confirmation
                            await ws.send_json({
                                "type": "pdf_uploaded",
                                "filename": filename,
                                "chunks": result.get("chunks", 0),
                                "pages": result.get("pages", 0),
                                "status": result.get("status"),
                                "error": result.get("error")
                            })
                            
                            continue
                        
                        # Handle get PDF list
                        elif data.get("type") == "get_pdf_list":
                            pdf_list = pdf_store.get_pdf_list()
                            await ws.send_json({
                                "type": "pdf_list",
                                "pdfs": pdf_list
                            })
                            continue
                        
                        # Handle STOP command
                        elif msg["text"] == "STOP":
                            print("\nℹ️ Stop command received")
                            
                            if vad_processor:
                                final_segment = vad_processor.finalize()
                                
                                if final_segment:
                                    try:
                                        transcript = transcribe_audio(final_segment)
                                        
                                        hallucinations = ["thank you", "thanks", "bye", "goodbye", "you"]
                                        if transcript.lower().strip() not in hallucinations or len(transcript) >= 15:
                                            await ws.send_json({
                                                "type": "final_transcript",
                                                "text": transcript
                                            })
                                            
                                            if len(transcript) > 5:
                                                agent_response = process_with_agent(transcript, pdf_store)
                                                
                                                await ws.send_json({
                                                    "type": "agent_response",
                                                    "text": agent_response,
                                                    "is_partial": False
                                                })
                                    except Exception as e:
                                        print(f"   ❌ Error: {e}")
                            
                            break
                    
                    except json.JSONDecodeError:
                        # Handle plain text STOP
                        if msg["text"] == "STOP":
                            break

                # Handle audio bytes
                elif isinstance(msg.get("bytes"), bytes):
                    audio_bytes = msg["bytes"]
                    
                    if vad_processor:
                        speech_segments, speech_prob = vad_processor.add_audio(audio_bytes)
                        
                        await ws.send_json({
                            "type": "vad_update",
                            "probability": speech_prob
                        })
                        
                        for segment in speech_segments:
                            segment_count += 1
                            print(f"\n🎤 Processing speech segment #{segment_count}")
                            
                            try:
                                transcript = transcribe_audio(segment)
                                print(f"   Transcript: '{transcript}'")
                                
                                hallucinations = ["thank you", "thanks", "bye", "goodbye", "you"]
                                transcript_lower = transcript.lower().strip()
                                
                                if transcript_lower in hallucinations and len(transcript) < 15:
                                    print(f"   ⚠️ Skipping hallucination")
                                    continue
                                
                                await ws.send_json({
                                    "type": "partial_transcript",
                                    "text": transcript,
                                    "segment": segment_count
                                })

                                if len(transcript) > 5:
                                    agent_response = process_with_agent(transcript, pdf_store)

                                    await ws.send_json({
                                        "type": "agent_response",
                                        "text": agent_response,
                                        "is_partial": False
                                    })
                                
                            except Exception as e:
                                print(f"   ❌ Error: {e}")

    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pdf_store.clear()
        print("\n🔌 WebSocket Closed\n")


@app.get("/")
def read_root():
    return {
        "status": "Voice Agent Backend Running", 
        "websocket": "/ws/audio",
        "features": ["STT (Whisper)", "LLM Agent (Groq)", "VAD (Silero)", "PDF Q&A (FAISS)"],
        "tools": ["get_ip_address", "search_web", "get_college_info", "query_uploaded_pdf"],
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "components": {
            "groq_api": "ok" if os.getenv("GROQ_API_KEY") else "missing",
            "hf_api": "ok" if os.getenv("HF_API_KEY") else "missing",
            "vector_store": "loaded" if vectorstore else "not loaded",
            "vad_model": "loaded" if model else "not loaded"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)