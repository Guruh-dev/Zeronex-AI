import sentry_sdk
from config import CONFIG
from utils.logger import get_logger

log = get_logger()

def init_sentry():
    """
    Inisialisasi Sentry jika DSN disediakan dalam konfigurasi.
    """
    dsn = CONFIG.get("SENTRY_DSN")
    if dsn:
        try:
            sentry_sdk.init(
                dsn=dsn,
                traces_sample_rate=1.0  # Sesuaikan sample rate sesuai kebutuhan
            )
            log.info("Sentry telah diinisialisasi dengan DSN yang diberikan.")
        except Exception as e:
            log.error("Gagal menginisialisasi Sentry: %s", str(e))
    else:
        log.info("SENTRY_DSN tidak ditemukan. Sentry tidak diinisialisasi.")

def capture_exception(error):
    """
    Meneruskan error ke Sentry.
    """
    try:
        sentry_sdk.capture_exception(error)
        log.info("Error telah dilaporkan ke Sentry.")
    except Exception as e:
        log.error("Gagal mengirim error ke Sentry: %s", str(e))
        log.error("Error asli: %s", str(error))
        log.error("Stack trace: %s", error.__traceback__)
        log.error("Pastikan Sentry SDK telah diinisialisasi dengan benar.")
        