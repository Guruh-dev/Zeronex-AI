import logging
from config import CONFIG

def get_logger():
    logger = logging.getLogger("EnhancedAI")
    level = getattr(logging, CONFIG.get("logging_level", "INFO"))
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
