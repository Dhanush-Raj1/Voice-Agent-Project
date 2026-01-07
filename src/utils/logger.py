import logging
import sys

def setup_logger(name: str = "voice_agent", level=logging.INFO):
    """Setup and return a logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)                        # logger level info 
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)    # send logs to the console
    handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Add handler if not already added
    if not logger.handlers:                       # setup_logger will be called multiple times in the code base, 
        logger.addHandler(handler)                # without this check, multiple handlers will be added, duplicate logs appear.
    
    return logger