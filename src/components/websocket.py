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
    """
    Handle WebSocket connection
    """
    
    await websocket.accept()               # accept the connection 
    logger.info("=" * 60)
    logger.info("WebSocket Connected")
    logger.info("=" * 60)


    # Create session-specific stores, 
    # each websocket client gets its own VAD, pdf memory, speech segment count
    # each time a client connects, these are re-initialized, when client disconnects, they are cleared
    vad_processor = SileroVADProcessor(vad_model, vad_utils) if vad_model else None
    pdf_store = SessionPDFStore(embeddings)
    segment_count = 0

    try:
        while True:
            msg = await websocket.receive()                 # wait for a message from the client 

            if msg["type"] == "websocket.disconnect":       # if client disconnects loop breaks and clean up happens 
                break                                        

            if msg["type"] == "websocket.receive":          # if client message is received 

                # Handle JSON messages - can only be "upload_pdf" or "STOP"
                if "text" in msg:
                    try:
                        data = json.loads(msg["text"])
                        
                        # 1. PDF upload
                        # the user uploads a pdf, it is preprocessed but nothing happens
                        if data.get("type") == "upload_pdf":
                            filename = data.get("filename")
                            pdf_base64 = data.get("data")
                            
                            logger.info(f"Receiving PDF upload: {filename}")
                                                                                  # WebSockets can only send text or binary, and browsers convert files to Base64 text
                            pdf_bytes = base64.b64decode(pdf_base64)              # therefore decode Base64 back to raw bytes(raw bytes - valid pdf data)
                            result = pdf_store.add_pdf(filename, pdf_bytes)
                            
                            await websocket.send_json({"type": "pdf_uploaded",
                                                     "filename": filename,
                                                     "chunks": result.get("chunks", 0),
                                                     "pages": result.get("pages", 0),
                                                     "status": result.get("status"),
                                                     "error": result.get("error")})
                            continue
                        
                        # 2. STOP command
                        # user clicks "Stop" in the UI, VAD finalizes, transcribes and then response is generated
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
                                            await websocket.send_json({"type": "final_transcript",
                                                                       "text": transcript})
                                            
                                            if len(transcript) > 5:
                                                agent_response = process_with_agent(transcript, pdf_store)

                                                # Generate TTS audio
                                                audio_data = await text_to_speech(agent_response)
                                                
                                                await websocket.send_json({"type": "agent_response",
                                                                            "text": agent_response,
                                                                            # audio is binary, we are sending json, convert binary to base64 text to fit inside json
                                                                            "audio": base64.b64encode(audio_data).decode('utf-8') if audio_data else None,           
                                                                            "is_partial": False})
                                    except Exception as e:
                                        logger.error(f"Error: {e}")
                            
                            break  # jumps to finally (at the end)
                    
                    # if text message is NOT valid JSON, and that text is "STOP", then exit the WebSocket loop(while True loop)
                    # and end the session
                    except json.JSONDecodeError:
                        if msg["text"] == "STOP":
                            break


                # Handle audio bytes
                # user talks, capture the audio, transfer it to VAD, transcribe it, process it with agent and response with TTS 
                elif isinstance(msg.get("bytes"), bytes):
                    audio_bytes = msg["bytes"]
                    
                    if vad_processor:
                        speech_segments, speech_prob = vad_processor.add_audio(audio_bytes)
                        
                        await websocket.send_json({"type": "vad_update",
                                                   "probability": speech_prob})
                        
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
                                
                                await websocket.send_json({"type": "partial_transcript",
                                                           "text": transcript,
                                                           "segment": segment_count})

                                if len(transcript) > 5:
                                    agent_response = process_with_agent(transcript, pdf_store)

                                    # Generate TTS audio 
                                    audio_data = await text_to_speech(agent_response)

                                    await websocket.send_json({"type": "agent_response",
                                                               "text": agent_response,
                                                               "audio": base64.b64encode(audio_data).decode('utf-8') if audio_data else None,
                                                               "is_partial": False})
                                
                            except Exception as e:
                                logger.error(f"Error: {e}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    finally:
        pdf_store.clear()
        logger.info("WebSocket Closed")