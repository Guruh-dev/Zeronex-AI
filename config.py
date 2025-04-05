import os
from utils import load_dotenv

# Memuat variabel lingkungan dari file .env (jika ada)
load_dotenv()

CONFIG = {
    # Konfigurasi model dan input
    "MODEL_NAME": os.getenv("MODEL_NAME", "deepset/xlm-roberta-large-squad2"),
    "SPEECH_LANGUAGE": os.getenv("SPEECH_LANGUAGE", "id-ID"),
    "MAX_CONVERSATION_HISTORY": int(os.getenv("MAX_CONVERSATION_HISTORY", 20)),
    
    # Konfigurasi logging
    "LOGGING_LEVEL": os.getenv("LOGGING_LEVEL", "INFO"),
    
    # Konfigurasi lingkungan (development atau production)
    "ENVIRONMENT": os.getenv("ENVIRONMENT", "development"),  # "development" atau "production"
    
    # Konfigurasi notifikasi error via email
    "ERROR_NOTIFICATION": {
        "EMAIL_ENABLED": os.getenv("EMAIL_ENABLED", "False").lower() in ("true", "1"),
        "EMAIL_HOST": os.getenv("EMAIL_HOST", ""),
        "EMAIL_PORT": int(os.getenv("EMAIL_PORT", 587)),
        "EMAIL_USER": os.getenv("EMAIL_USER", ""),
        "EMAIL_PASSWORD": os.getenv("EMAIL_PASSWORD", ""),
        "ADMIN_EMAILS": os.getenv("ADMIN_EMAILS", "").split(",") if os.getenv("ADMIN_EMAILS") else [],
    },
    
    # Konfigurasi integrasi dengan Sentry (jika digunakan)
    "SENTRY_DSN": os.getenv("SENTRY_DSN", None),
    
    # Konfigurasi retry untuk operasi kritis
    "RETRY_SETTINGS": {
        "MAX_RETRIES": int(os.getenv("MAX_RETRIES", 3)),
        "RETRY_DELAY": int(os.getenv("RETRY_DELAY", 2)),  # delay dalam detik
    }
}
