import edge_tts
from src.utils.logger import setup_logger

logger = setup_logger("tts")


VOICE = "en-US-AriaNeural"  
# Alternatives:
# "en-US-GuyNeural"     # Male, professional
# "en-US-JennyNeural"   # Female, warm
# "en-GB-SoniaNeural"   # British female

async def text_to_speech(text: str, voice: str = VOICE) -> bytes:
    """Convert text to speech using Edge TTS"""

    try:
        logger.info(f"Converting text to speech: '{text[:50]}...'")
        
        # creating TTS streaming session, does not geneate audio yet
        communicate = edge_tts.Communicate(text, voice)
         
        audio_bytes = b""   # empty binary buffer 

        # stream audio chunks
        async for chunk in communicate.stream():           # tts streams audios and delivers in chunks 
            if chunk["type"] == "audio":                   # stream sends different chunk types, "audio", "word", "metadata"
                audio_bytes += chunk["data"]               # append audio 
        
        logger.info(f"✅ Generated {len(audio_bytes)} bytes of audio")
        return audio_bytes
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return b""