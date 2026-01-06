import json
import base64
from fastapi import WebSocket
from components.audio import SileroVADProcessor, transcribe_audio
from components.pdf_handler import SessionPDFStore
from components.agent import process_with_agent
from utils.logger import setup_logger

logger = setup_logger("websocket")


async def handle_websocket(
    websocket: WebSocket,
    vad_model,
    vad_utils,
    embeddings
):
    """Handle WebSocket connection"""
    await websocket.accept()
    logger.info("=" * 60)
    logger.info("WebSocket Connected")
    logger.info("=" * 60)

    # Create session-specific stores
    vad_processor = SileroVADProcessor(vad_model, vad_utils) if vad_model else None
    pdf_store = SessionPDFStore(embeddings)
    segment_count = 0

    try:
        while True:
            msg = await websocket.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if msg["type"] == "websocket.receive":
                # Handle JSON messages
                if "text" in msg:
                    try:
                        data = json.loads(msg["text"])
                        
                        # PDF upload
                        if data.get("type") == "upload_pdf":
                            filename = data.get("filename")
                            pdf_base64 = data.get("data")
                            
                            logger.info(f"Receiving PDF upload: {filename}")
                            
                            pdf_bytes = base64.b64decode(pdf_base64)
                            result = pdf_store.add_pdf(filename, pdf_bytes)
                            
                            await websocket.send_json({
                                "type": "pdf_uploaded",
                                "filename": filename,
                                "chunks": result.get("chunks", 0),
                                "pages": result.get("pages", 0),
                                "status": result.get("status"),
                                "error": result.get("error")
                            })
                            continue
                        
                        # STOP command
                        elif msg["text"] == "STOP":
                            logger.info("Stop command received")
                            
                            if vad_processor:
                                final_segment = vad_processor.finalize()
                                
                                if final_segment:
                                    try:
                                        transcript = transcribe_audio(final_segment)
                                        
                                        # Filter hallucinations
                                        hallucinations = ["thank you", "thanks", "bye", "goodbye", "you"]
                                        if transcript.lower().strip() not in hallucinations or len(transcript) >= 15:
                                            await websocket.send_json({
                                                "type": "final_transcript",
                                                "text": transcript
                                            })
                                            
                                            if len(transcript) > 5:
                                                agent_response = process_with_agent(transcript, pdf_store)
                                                
                                                await websocket.send_json({
                                                    "type": "agent_response",
                                                    "text": agent_response,
                                                    "is_partial": False
                                                })
                                    except Exception as e:
                                        logger.error(f"Error: {e}")
                            
                            break
                    
                    except json.JSONDecodeError:
                        if msg["text"] == "STOP":
                            break

                # Handle audio bytes
                elif isinstance(msg.get("bytes"), bytes):
                    audio_bytes = msg["bytes"]
                    
                    if vad_processor:
                        speech_segments, speech_prob = vad_processor.add_audio(audio_bytes)
                        
                        await websocket.send_json({
                            "type": "vad_update",
                            "probability": speech_prob
                        })
                        
                        for segment in speech_segments:
                            segment_count += 1
                            logger.info(f"Processing speech segment #{segment_count}")
                            
                            try:
                                transcript = transcribe_audio(segment)
                                logger.info(f"Transcript: '{transcript}'")
                                
                                # Filter hallucinations
                                hallucinations = ["thank you", "thanks", "bye", "goodbye", "you"]
                                if transcript.lower().strip() in hallucinations and len(transcript) < 15:
                                    logger.info("Skipping hallucination")
                                    continue
                                
                                await websocket.send_json({
                                    "type": "partial_transcript",
                                    "text": transcript,
                                    "segment": segment_count
                                })

                                if len(transcript) > 5:
                                    agent_response = process_with_agent(transcript, pdf_store)

                                    await websocket.send_json({
                                        "type": "agent_response",
                                        "text": agent_response,
                                        "is_partial": False
                                    })
                                
                            except Exception as e:
                                logger.error(f"Error: {e}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    finally:
        pdf_store.clear()
        logger.info("WebSocket Closed")