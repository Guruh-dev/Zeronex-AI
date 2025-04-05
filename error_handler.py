import logging
import traceback
from utils.logger import get_logger
from config import CONFIG
from EnhancedAI.utils.sentry_sdk import capture_exception

log = get_logger()

def send_email_alert(subject, message):
    """
    Mengirim notifikasi email kepada admin apabila terjadi error kritis.
    Pastikan konfigurasi email telah diatur dengan benar di config.py.
    """
    email_config = CONFIG.get("ERROR_NOTIFICATION", {})
    if not email_config.get("EMAIL_ENABLED", False):
        log.debug("Notifikasi email tidak diaktifkan.")
        return
    try:
        import smtplib
        from email.mime.text import MIMEText

        smtp_server = email_config.get("EMAIL_HOST")
        smtp_port = email_config.get("EMAIL_PORT")
        email_user = email_config.get("EMAIL_USER")
        email_password = email_config.get("EMAIL_PASSWORD")
        admin_emails = email_config.get("ADMIN_EMAILS", [])

        if not (smtp_server and email_user and email_password and admin_emails):
            log.error("Konfigurasi email tidak lengkap, tidak dapat mengirim notifikasi.")
            return

        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = email_user
        msg["To"] = ", ".join(admin_emails)

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_user, email_password)
            server.sendmail(email_user, admin_emails, msg.as_string())
        log.info("Email notifikasi error terkirim dengan sukses.")
    except Exception as e:
        log.error("Gagal mengirim notifikasi email: %s", str(e))

def handle_error(error, notify=True):
    """
    Menangani error dengan melakukan logging lengkap error beserta stack trace.
    Jika lingkungan adalah production dan notifikasi diaktifkan, akan mengirim email notifikasi.
    Juga meneruskan error ke Sentry melalui modul sentry_integration.
    """
    error_message = "".join(traceback.format_exception(None, error, error.__traceback__))
    log.error("Terjadi kesalahan:\n%s", error_message)

    # Integrasi dengan Sentry melalui modul sentry_integration
    if CONFIG.get("SENTRY_DSN"):
        capture_exception(error)

    # Mengirim notifikasi email apabila di lingkungan production
    if notify and CONFIG.get("ENVIRONMENT", "development") == "production":
        subject = "Critical Error in EnhancedAI System"
        send_email_alert(subject, error_message)

def error_handler_decorator(func):
    """
    Decorator untuk membungkus fungsi agar error yang terjadi dapat ditangani secara otomatis.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            handle_error(e)
            raise e
    return wrapper
