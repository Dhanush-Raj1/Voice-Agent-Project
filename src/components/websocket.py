import json
import base64
from fastapi import WebSocket
from src.components.tts import text_to_speech
from src.components.audio import SileroVADProcessor, transcribe_audio
from src.components.pdf_handler import SessionPDFStore
from src.components.agent import process_with_agent
from src.utils.logger import setup_logger

logger = setup_logger("websocket")



# connection to the client
async def handle_websocket(websocket: WebSocket, vad_model, vad_utils, embeddings):
    await websocket.accept()
    logger.info("=" * 60)
    logger.info("WebSocket Connected")
    logger.info("=" * 60)

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
                        
                        # 1. PDF upload
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
                        
                        # 2. STOP command - NEW VERSION
                        elif data.get("type") == "stop":
                            logger.info("⏸️ Stop command received (pause mode)")
                            
                            if vad_processor:
                                final_segment = vad_processor.finalize()
                                
                                if final_segment:
                                    try:
                                        transcript = transcribe_audio(final_segment)
                                        hallucinations = ["thank you", "thanks", "bye", "goodbye", "you"]

                                        if transcript.lower().strip() not in hallucinations or len(transcript) >= 15:
                                            await websocket.send_json({
                                                "type": "final_transcript",
                                                "text": transcript
                                            })
                                            
                                            if len(transcript) > 5:
                                                agent_response = process_with_agent(transcript, pdf_store)
                                                audio_data = await text_to_speech(agent_response)
                                                
                                                await websocket.send_json({
                                                    "type": "agent_response",
                                                    "text": agent_response,
                                                    "audio": base64.b64encode(audio_data).decode('utf-8') if audio_data else None,
                                                    "is_partial": False
                                                })
                                        else:
                                            logger.info("Skipped hallucination, sending acknowledgment")
                                            # Still send acknowledgment even if skipped
                                            await websocket.send_json({
                                                "type": "stop_acknowledged",
                                                "message": "Ready for next recording"
                                            })
                                    except Exception as e:
                                        logger.error(f"Error processing final segment: {e}")
                                        # Send acknowledgment even on error
                                        await websocket.send_json({
                                            "type": "stop_acknowledged",
                                            "message": "Ready for next recording"
                                        })
                                else:
                                    # No final segment, just acknowledge
                                    await websocket.send_json({
                                        "type": "stop_acknowledged",
                                        "message": "Ready for next recording"
                                    })
                                
                                # Reset VAD for next recording session
                                vad_processor.reset()
                                segment_count = 0
                                logger.info("✅ VAD reset, ready for next recording")
                            else:
                                # No VAD processor, just acknowledge
                                await websocket.send_json({
                                    "type": "stop_acknowledged",
                                    "message": "Ready for next recording"
                                })
                            
                            # Continue the loop, don't break!
                            continue
                    
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON: {msg['text']}")
                        continue

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
                                    audio_data = await text_to_speech(agent_response)

                                    await websocket.send_json({
                                        "type": "agent_response",
                                        "text": agent_response,
                                        "audio": base64.b64encode(audio_data).decode('utf-8') if audio_data else None,
                                        "is_partial": False
                                    })
                                
                            except Exception as e:
                                logger.error(f"Error processing segment: {e}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    finally:
        pdf_store.clear()
        logger.info("WebSocket Closed")