import io
import torch
import numpy as np
import soundfile as sf
from groq import Groq
from src.utils.config import SAMPLE_RATE, VAD_THRESHOLD, MIN_SPEECH_DURATION_MS, MIN_SILENCE_DURATION_MS, GROQ_API_KEY
from src.utils.logger import setup_logger

logger = setup_logger("audio")
groq_client = Groq(api_key=GROQ_API_KEY)



class SileroVADProcessor:
    """Voice Activity Detection using Silero VAD"""
    
    def __init__(self, model, utils_func):
        self.model = model
        self.get_speech_timestamps = utils_func
        self.reset()
    

    def reset(self):
        self.audio_buffer = []
        self.speech_buffer = []
        self.is_speech_active = False
        self.last_speech_prob = 0.0
        self.silence_frames = 0
        self.speech_frames = 0
        

    def add_audio(self, audio_bytes):
        """
        Add audio data and return speech segments when detected
        """

        # Convert bytes - float audio
        # Web audio comes as 16-bit PCM, VAD expects float32.
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)      # convert PCM bytes into int16 audio samples 
        audio_float32 = audio_int16.astype(np.float32) / 32768.0      # and then convert to float 32 and normalize 
                                                                       # (Silero VAD, Whisper) expects float audio in [-1, 1]
        
        # stores continuous audio samples.
        self.audio_buffer.extend(audio_float32)    
        
        segments = []
        frame_size = 512    # Silero VAD is trained on 512-sample frames
        
        while len(self.audio_buffer) >= frame_size:               # processes audio only when enough samples exist to form one VAD frame
            frame = self.audio_buffer[:frame_size]                    # each frame should be 512 
            self.audio_buffer = self.audio_buffer[frame_size:]
            
            try:
                audio_tensor = torch.from_numpy(np.array(frame, dtype=np.float32))     # converts 512 audio into pytorch tensor
                speech_prob = self.model(audio_tensor, SAMPLE_RATE).item()             # feed audio frame to silero, output is probabilty that this frame contains speech 
                self.last_speech_prob = speech_prob                                    # .item() converts torch scalar tensor to float 
                
                if speech_prob >= VAD_THRESHOLD:                # considered speech if speech_prob > 0.5 
                    self.speech_frames += 1
                    self.silence_frames = 0
                    
                    if not self.is_speech_active:
                        self.is_speech_active = True
                        logger.info(f"Speech started (prob: {speech_prob:.3f})")
                    
                    self.speech_buffer.extend(frame)           # collect raw audio samples 
                    
                else:                                         # speech_prob < 0.5 
                    if self.is_speech_active:                 # we were speacking, now silence is appearing    
                        self.silence_frames += 1
                        self.speech_buffer.extend(frame)
                        
                        speech_duration_ms = (self.speech_frames * frame_size / SAMPLE_RATE) * 1000
                        silence_duration_ms = (self.silence_frames * frame_size / SAMPLE_RATE) * 1000
                        
                        if (silence_duration_ms >= MIN_SILENCE_DURATION_MS and           # speech only ends if silence is longer than min silence duration
                            speech_duration_ms >= MIN_SPEECH_DURATION_MS):               # and speech is longer than min speech duration
                            
                            logger.info(f"Speech ended (duration: {speech_duration_ms:.0f}ms)")
                            
                            # Whisper STT expects int16 PCM 
                            speech_int16 = (np.array(self.speech_buffer) * 32768.0).astype(np.int16)      # convert back to int16 PCM 
                            segment_bytes = speech_int16.tobytes()                                        # reverse earlier normalization
                                                                                                        
                            segments.append(segment_bytes)        # complete speech chunk ready for transcription
                            
                            # reset for next speech 
                            self.speech_buffer = []
                            self.is_speech_active = False
                            self.silence_frames = 0
                            self.speech_frames = 0

                    else:                                         # silence before any speech 
                        self.silence_frames += 1        
                    
            except Exception as e:
                logger.error(f"VAD processing error: {e}")
        
        return segments, self.last_speech_prob
    

    def finalize(self):
        """Force end current speech segment if active"""

        if self.is_speech_active and len(self.speech_buffer) > 0:
            speech_duration_ms = (len(self.speech_buffer) / SAMPLE_RATE) * 1000
            
            if speech_duration_ms >= MIN_SPEECH_DURATION_MS:
                logger.info(f"Finalizing speech segment ({speech_duration_ms:.0f}ms)")
                
                speech_int16 = (np.array(self.speech_buffer) * 32768.0).astype(np.int16)
                segment_bytes = speech_int16.tobytes()
                
                self.reset()
                return segment_bytes
        
        self.reset()
        return None





def load_vad_model():
    """Load Silero VAD model"""

    logger.info("Loading Silero VAD model...")
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      onnx=False)
        
        (get_speech_timestamps, _, read_audio, *_) = utils
        
        logger.info(f"✅ Silero VAD loaded (threshold: {VAD_THRESHOLD})")
        return model, get_speech_timestamps
        
    except Exception as e:
        logger.error(f"Silero VAD loading failed: {e}")
        return None, None         # websocket won't crash but audio processing will be disabled



def transcribe_audio(audio_bytes):
    """Convert audio bytes to transcript using Groq Whisper"""

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    audio_np /= 32768.0
    
    buffer = io.BytesIO()                                     # Create in-memory WAV file
    sf.write(buffer, audio_np, SAMPLE_RATE, format='WAV')     # Write WAV file, because Whisper API expects an audio file 
    buffer.seek(0)                                            # without this file would appear empty
    buffer.name = "audio.wav"                                 # why a buffer name is needed, Groq API expects a filename
    
    transcription = groq_client.audio.transcriptions.create(
                        file=buffer,
                        model="whisper-large-v3-turbo",
                        language="en",
                        temperature=0.0,                     # no hallucinations, zero creativity
                        response_format="text"
                    )
    
    return transcription.strip()