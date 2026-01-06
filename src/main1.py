# import uvicorn
# from fastapi import FastAPI, WebSocket
# from fastapi.middleware.cors import CORSMiddleware

# from components.websocket import handle_websocket
# from components.audio import load_vad_model
# from components.vectorstore import load_college_vectorstore
# from components.tools import set_college_vectorstore
# from utils.logger import setup_logger

# logger = setup_logger("main")

# app = FastAPI(title="Voice + RAG Agent")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# vad_model = None
# vad_utils = None
# college_vectorstore = None
# embeddings = None

# @app.on_event("startup")
# async def startup_event():
#     global vad_model, vad_utils, college_vectorstore, embeddings

#     logger.info("=" * 60)
#     logger.info("🚀 Starting Voice Agent Server")
#     logger.info("=" * 60)

#     # Load Silero VAD
#     vad_model, vad_utils = load_vad_model()

#     # Load college vectorstore (Pinecone)
#     college_vectorstore, embeddings = load_college_vectorstore()

#     # Make vectorstore available to tools
#     if college_vectorstore:
#         set_college_vectorstore(college_vectorstore)

#     logger.info("✅ Startup completed successfully")


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await handle_websocket(
#         websocket=websocket,
#         vad_model=vad_model,
#         vad_utils=vad_utils,
#         embeddings=embeddings
#     )


# @app.get("/")
# def read_root():
#     return {
#         "status": "Voice Agent Backend Running", 
#         "websocket": "/ws/audio",
#         "features": ["STT (Whisper)", "LLM Agent (Groq)", "VAD (Silero)", "PDF Q&A (FAISS)"],
#         "tools": ["get_ip_address", "search_web", "get_college_info", "query_uploaded_pdf"],
#     }


# @app.get("/health")
# def health():
#     return {"status": "ok"}



# if __name__ == "__main__":
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True,
#         log_level="info"
#     )



import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from components.websocket import handle_websocket
from components.audio import load_vad_model
from components.vectorstore import load_college_vectorstore
from components.tools import set_college_vectorstore
from utils.logger import setup_logger

logger = setup_logger("main")

# ---------------------------------------------------------
# Global resources
# ---------------------------------------------------------
vad_model = None
vad_utils = None
college_vectorstore = None
embeddings = None


# ---------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vad_model, vad_utils, college_vectorstore, embeddings

    logger.info("=" * 60)
    logger.info("🚀 Starting Voice Agent Server")
    logger.info("=" * 60)

    # Load Silero VAD
    vad_model, vad_utils = load_vad_model()

    # Load Pinecone college vectorstore
    college_vectorstore, embeddings = load_college_vectorstore()

    # Make vectorstore available to tools
    if college_vectorstore:
        set_college_vectorstore(college_vectorstore)

    logger.info("✅ Startup completed successfully")

    yield  # ---- App is running ----

    # Shutdown logic (optional)
    logger.info("🛑 Shutting down Voice Agent Server")


# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------
app = FastAPI(
    title="Voice + RAG Agent",
    lifespan=lifespan
)

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await handle_websocket(
        websocket=websocket,
        vad_model=vad_model,
        vad_utils=vad_utils,
        embeddings=embeddings
    )


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------
# Run server
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
