import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path

from src.components.websocket import handle_websocket
from src.components.audio import load_vad_model
from src.components.vectorstore import load_college_vectorstore
from src.components.tools import set_college_vectorstore
from src.utils.logger import setup_logger

logger = setup_logger("main")

# Global resources, loaded onces, shared by websocket clients 
vad_model = None
vad_utils = None
college_vectorstore = None
embeddings = None


# Lifespan (startup / shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vad_model, vad_utils, college_vectorstore, embeddings     # allows assigning the global resources 

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

    yield               # starts the app, server is running

    logger.info("🛑 Shutting down Voice Agent Server")



app = FastAPI(title="Voice + RAG Agent",
              lifespan=lifespan)



# CORS - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the index.html file"""

    html_path = Path(__file__).parent.parent / "templates" / "index.html"
    
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(
            content="<h1>Frontend not found</h1><p>Please ensure templates/index.html exists</p>",
            status_code=404
        )


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for voice agent"""

    try:
        await handle_websocket(
            websocket=websocket,
            vad_model=vad_model,
            vad_utils=vad_utils,
            embeddings=embeddings
        )
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "vad_loaded": vad_model is not None,
        "vectorstore_loaded": college_vectorstore is not None
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Changed from "main:app" to "main1:app"
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )